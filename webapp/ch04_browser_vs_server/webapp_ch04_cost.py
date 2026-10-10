"""기능을 서버로 옮기면 요청 수·대기 시간·서버비가 어떻게 바뀌는지 세는 모형.

- 대기 시간: 왕복 지연(RTT) + 올리는 시간 + 서버 계산 + 서버가 기다리는 시간 + 내려받는 시간.
- Vercel: Active CPU(코드가 실제로 도는 시간), Provisioned Memory(메모리 × 인스턴스가 살아 있는 시간),
  Invocations(요청 수), Fast Origin Transfer(CDN과 함수 사이 바이트). CDN 요청과 방문자 쪽 전송은
  Pro에 들어 있는 Flat Rate CDN 등급(100만 요청, 1TB)으로 덮고, 넘으면 다음 등급 값을 더한다.
  요청 하나가 인스턴스 하나를 혼자 쓴다고 둬서 메모리 값은 위쪽 어림이다(실제로는 여러 요청이 나눠 쓴다).
- Netlify: 크레딧. 계산 = 메모리(GB) × 함수가 돈 전체 시간, 요청 1만 개당 2, 전송 1GB당 20.

돈은 모두 분수(Fraction)로 세고, 결과를 적을 때만 반올림한다. 시간 단위는 밀리초(ms).
"""

import json
import math
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
PRICES = HERE / "data" / "prices_2026-10-10.json"
ASSUME = HERE / "data" / "assumptions.json"


def load():
    return (json.loads(PRICES.read_text(encoding="utf-8")),
            json.loads(ASSUME.read_text(encoding="utf-8")))


# ---------------- 타자 치는 모습과 디바운스 ----------------

def keystroke_times(t):
    """묶음(bursts)마다 keys_per_burst번, 묶음 안 간격 key_gap_ms, 묶음 사이 burst_gap_ms."""
    times, now = [], 0
    for b in range(t["bursts"]):
        for k in range(t["keys_per_burst"]):
            times.append(now)
            if k < t["keys_per_burst"] - 1:
                now += t["key_gap_ms"]
        now += t["burst_gap_ms"]
    return times


def debounced_requests(times, wait_ms):
    """마지막 입력 뒤 wait_ms 동안 새 입력이 없을 때만 요청 하나를 보낸다. 보낸 시각 목록."""
    sent = []
    for i, t in enumerate(times):
        nxt = times[i + 1] if i + 1 < len(times) else None
        if nxt is None or nxt - t >= wait_ms:
            sent.append(t + wait_ms)
    return sent


# ---------------- 대기 시간 ----------------

def transfer_ms(nbytes, mbps):
    return F(nbytes * 8, mbps * 1_000_000) * 1000


def server_latency_ms(job, net, region):
    return (F(net["rtt_ms"][region]) + transfer_ms(job["request_bytes"], net["up_mbps"])
            + F(job["cpu_ms"]) + F(job.get("wait_ms", 0)) + transfer_ms(job["response_bytes"], net["down_mbps"]))


def latency_table(a, region="icn1"):
    net, s, b = a["network"], a["server"], a["browser"]
    deb = F(a["typing"]["debounce_ms"])
    return {
        "preview_browser": F(b["preview_ms"]),
        "preview_server": server_latency_ms(s["preview"], net, region),
        "preview_server_debounced_after_last_key": deb + server_latency_ms(s["preview"], net, region),
        "export_browser": F(b["export_ms"]),
        "export_server": server_latency_ms(s["export_png"], net, region),
        "ai_background_server": server_latency_ms(s["ai_background"], net, region),
        "archive_save_server": server_latency_ms(s["archive_save"], net, region),
        "draft_save_browser": F(b["draft_save_ms"]),
    }


# ---------------- 한 장 만들 때 오가는 요청 ----------------

DESIGNS = {
    "png_in_browser": "AI 배경만 서버, 최종 PNG까지 브라우저",
    "decided": "판단표대로(AI 배경·최종 PNG·보관 저장은 서버)",
    "server_preview": "판단표 + 미리보기도 서버(키마다 요청)",
    "server_preview_debounced": "판단표 + 미리보기도 서버(300ms 디바운스)",
}


def card_requests(a, design):
    """카드 한 장을 만들 때의 요청 목록. 항목: (무엇, 종류, 횟수, job 또는 None, 요청 바이트, 응답 바이트)."""
    s = a["server"]
    # page_open.bytes는 요청 8개의 합이라, 요청 하나의 평균(분수)으로 넣어야 횟수를 곱했을 때 합이 맞는다.
    po = a["page_open"]
    reqs = [("페이지 열기", "static", po["requests"], None, 0, F(po["bytes"], po["requests"])),
            ("템플릿 바꾸기", "static", 3, None, 0, a["template_change"]["bytes"]),
            ("AI 배경 생성", "function", 2, s["ai_background"], s["ai_background"]["request_bytes"], s["ai_background"]["response_bytes"]),
            ("보관 저장", "function", 1, s["archive_save"], s["archive_save"]["request_bytes"], s["archive_save"]["response_bytes"])]
    if design != "png_in_browser":
        reqs.append(("최종 PNG 출력", "function", 1, s["export_png"], s["export_png"]["request_bytes"], s["export_png"]["response_bytes"]))
    if design == "server_preview":
        n = len(keystroke_times(a["typing"]))
        reqs.append(("미리보기", "function", n, s["preview"], s["preview"]["request_bytes"], s["preview"]["response_bytes"]))
    if design == "server_preview_debounced":
        n = len(debounced_requests(keystroke_times(a["typing"]), a["typing"]["debounce_ms"]))
        reqs.append(("미리보기", "function", n, s["preview"], s["preview"]["request_bytes"], s["preview"]["response_bytes"]))
    return reqs


def usage_per_card(a, design):
    u = {"requests": 0, "function_requests": 0, "cpu_ms": F(0), "wall_ms": F(0),
         "bytes_to_visitor": 0, "bytes_from_visitor": 0, "function_bytes": 0}
    for _name, kind, n, job, rq, rs in card_requests(a, design):
        u["requests"] += n
        u["bytes_to_visitor"] += n * rs
        u["bytes_from_visitor"] += n * rq
        if kind == "function":
            u["function_requests"] += n
            u["cpu_ms"] += n * F(job["cpu_ms"])
            u["wall_ms"] += n * (F(job["cpu_ms"]) + F(job.get("wait_ms", 0)))
            u["function_bytes"] += n * (rq + rs)
    return u


def monthly_usage(a, design):
    cards = a["traffic"]["users_per_month"] * a["traffic"]["cards_per_user_per_month"]
    u = usage_per_card(a, design)
    return {k: v * cards for k, v in u.items()} | {"cards": cards}


# ---------------- Vercel ----------------

def vercel_cost(p, a, m, region="icn1"):
    v = p["vercel"]
    r = v["regions"][region]
    gb = a["gb_bytes"]
    mem_gb = v["default_memory_gb"]
    cpu_h = m["cpu_ms"] / F(3_600_000)
    mem_gbh = mem_gb * m["wall_ms"] / F(3_600_000)
    parts = {
        "active_cpu": cpu_h * F(str(r["active_cpu_usd_per_hour"])),
        "provisioned_memory": mem_gbh * F(str(r["memory_usd_per_gb_hour"])),
        "invocations": F(m["function_requests"]) * F(str(v["invocation_usd_per_million"])) / 1_000_000,
    }
    if "fast_origin_transfer_usd_per_gb" in r:
        parts["fast_origin_transfer"] = F(m["function_bytes"], gb) * F(str(r["fast_origin_transfer_usd_per_gb"]))
    usage = sum(parts.values())
    transfer_tb = F(m["bytes_to_visitor"] + m["bytes_from_visitor"], gb * 1000)
    tier = next(t for t in v["flat_rate_cdn"]["tiers_usd_month"]
                if m["requests"] <= t["cdn_requests"] and transfer_tb <= t["data_transfer_tb"])
    over_credit = max(F(0), usage - v["pro_usage_credit_usd_month"])
    total = F(v["pro_platform_fee_usd_month"]) + over_credit + tier["price"]
    return {"cpu_hours": cpu_h, "memory_gb_hours": mem_gbh, "parts": parts, "usage": usage,
            "cdn_requests": m["requests"], "transfer_tb": transfer_tb, "cdn_tier_usd": tier["price"],
            "over_credit": over_credit, "bill": total}


def hobby_check(p, a, m):
    h = p["vercel"]["hobby"]["included"]
    gb = a["gb_bytes"]
    mem_gb = p["vercel"]["default_memory_gb"]
    used = {
        "active_cpu_hours": m["cpu_ms"] / F(3_600_000),
        "provisioned_memory_gb_hours": mem_gb * m["wall_ms"] / F(3_600_000),
        "invocations": F(m["function_requests"]),
        "cdn_requests": F(m["requests"]),
        "fast_data_transfer_gb": F(m["bytes_to_visitor"] + m["bytes_from_visitor"], gb),
        "fast_origin_transfer_gb": F(m["function_bytes"], gb),
    }
    return {k: {"used": used[k], "included": h[k], "over": used[k] > h[k]} for k in h}


# ---------------- Netlify ----------------

def netlify_credits(p, a, m):
    n = p["netlify"]
    gb = a["gb_bytes"]
    compute = n["default_function_memory_gb"] * m["wall_ms"] / F(3_600_000) * n["credits_per_gb_hour_compute"]
    requests = F(m["requests"], 10_000) * n["credits_per_10k_requests"]
    bandwidth = F(m["bytes_to_visitor"], gb) * n["credits_per_gb_bandwidth"]
    total = compute + requests + bandwidth
    plan = next((pl for pl in n["plans"] if total <= pl["credits"]), None)
    if plan:
        usd, name = F(plan["usd_month"]), plan["name"]
    else:
        pro = n["plans"][-1]
        rc = n["recharge"]["Pro"]
        packs = math.ceil((total - pro["credits"]) / rc["credits"])
        usd, name = F(pro["usd_month"] + packs * rc["usd"]), f"Pro + 충전 {packs}번"
    return {"compute": compute, "requests": requests, "bandwidth": bandwidth, "total": total,
            "plan": name, "usd": usd}


def ai_model_cost_krw(a, m):
    """AI 이미지 생성 단가(3장과 같은 가상 단가 1장 50원). 호스팅 비용과 따로 든다."""
    n_ai = m["cards"] * 2
    return n_ai * a["ai_model_cost_krw_per_image"]


if __name__ == "__main__":
    p, a = load()
    for d in DESIGNS:
        m = monthly_usage(a, d)
        c = vercel_cost(p, a, m)
        print(d, m["requests"], float(c["usage"]), float(c["bill"]))
