"""1인 개발 앱 출시 실전 4장 실험. `uv run python webapp/ch04_browser_vs_server/experiments.py` 로 실행한다.

E0 판단표: 기능 7개를 어디서 돌릴지(규칙과 이유)
E1 카드 한 장을 만들 때 오가는 요청: 설계 네 가지
E2 대기 시간: 브라우저와 서버(서울 icn1, 미국 동부 iad1)
E3 디바운스: 키 120번이 요청 몇 번이 되나
E4 요청 하나의 서버비: AI 배경 1번, 최종 PNG 1번, 서버 미리보기 1번(icn1)
E5 한 달 서버비(Vercel Pro, icn1): 설계 네 가지
E6 Vercel Hobby 포함량과 비교(Hobby는 비상업 개인용)
E7 Netlify 크레딧: 설계 네 가지
E8 AI 생성 단가와 호스팅비 비교(가상 단가)
E9 브라우저 저장소: localStorage에 들어가는 개수, base64, IndexedDB 한도, 지워지는 조건
E10 대조: 카드 한 장 값 × 장 수 = 한 달 값, 손계산 식 = 모형 값
E11 본문에 나오는 중간 계산값(전송 시간, 배수, 연습 답)

표준 라이브러리만 쓴다. 네트워크와 API 키 없이 돌고, 결과는 매번 같다.
"""

import json
import platform
import sys
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import webapp_ch04_cost as C  # noqa: E402
import webapp_ch04_decide as D  # noqa: E402
import webapp_ch04_storage as S  # noqa: E402

OUT = HERE / "results" / "ch04.json"


def r(x, nd=6):
    return round(float(x), nd)


def e1(a):
    out = {}
    for d, label in C.DESIGNS.items():
        rows = [{"what": n, "kind": k, "count": c, "request_bytes": rq, "response_bytes": r(rs, 3)}
                for n, k, c, _job, rq, rs in C.card_requests(a, d)]
        u = C.usage_per_card(a, d)
        out[d] = {"label": label, "rows": rows, "requests": u["requests"], "function_requests": u["function_requests"],
                  "bytes_to_visitor": int(u["bytes_to_visitor"])}
    return out


def e4(p, a):
    v = p["vercel"]
    reg = v["regions"]["icn1"]
    out = {}
    for name, job in a["server"].items():
        wall = F(job["cpu_ms"]) + F(job.get("wait_ms", 0))
        cpu = F(job["cpu_ms"], 3_600_000) * F(str(reg["active_cpu_usd_per_hour"]))
        mem = v["default_memory_gb"] * wall / 3_600_000 * F(str(reg["memory_usd_per_gb_hour"]))
        inv = F(str(v["invocation_usd_per_million"])) / 1_000_000
        fot = F(job["request_bytes"] + job["response_bytes"], a["gb_bytes"]) * F(str(reg["fast_origin_transfer_usd_per_gb"]))
        tot = cpu + mem + inv + fot
        out[name] = {"cpu_ms": job["cpu_ms"], "wall_ms": r(wall, 3), "bytes": job["request_bytes"] + job["response_bytes"],
                     "active_cpu_usd": r(cpu, 9), "memory_usd": r(mem, 9), "invocation_usd": r(inv, 9),
                     "origin_transfer_usd": r(fot, 9), "total_usd": r(tot, 9),
                     "total_krw": r(tot * a["usd_krw"], 4)}
    return out


def e5_e6_e7(p, a):
    e5, e6, e7 = {}, {}, {}
    for d in C.DESIGNS:
        m = C.monthly_usage(a, d)
        c = C.vercel_cost(p, a, m)
        e5[d] = {"cards": m["cards"], "requests": m["requests"], "function_requests": m["function_requests"],
                 "cpu_hours": r(c["cpu_hours"], 4), "memory_gb_hours": r(c["memory_gb_hours"], 4),
                 "parts_usd": {k: r(v, 4) for k, v in c["parts"].items()}, "usage_usd": r(c["usage"], 4),
                 "transfer_tb": r(c["transfer_tb"], 6), "cdn_tier_usd": c["cdn_tier_usd"],
                 "over_credit_usd": r(c["over_credit"], 4), "bill_usd": r(c["bill"], 4)}
        iad = C.vercel_cost(p, a, m, region="iad1")
        e5[d]["iad1_cpu_memory_usd"] = r(iad["parts"]["active_cpu"] + iad["parts"]["provisioned_memory"], 4)
        e5[d]["icn1_cpu_memory_usd"] = r(c["parts"]["active_cpu"] + c["parts"]["provisioned_memory"], 4)
        h = C.hobby_check(p, a, m)
        e6[d] = {k: {"used": r(v["used"], 4), "included": v["included"], "over": v["over"]} for k, v in h.items()}
        n = C.netlify_credits(p, a, m)
        e7[d] = {"compute": r(n["compute"], 4), "requests": r(n["requests"], 4), "bandwidth": r(n["bandwidth"], 4),
                 "total": r(n["total"], 4), "plan": n["plan"], "usd": r(n["usd"], 2)}
    return e5, e6, e7


def e9():
    txt = S.draft_text()
    key = "draft:card-0001"
    units = S.utf16_units(txt)
    fit, per = S.local_storage_fit(units, S.utf16_units(key))
    png = 1_500_000
    b64 = S.base64_chars(png)
    img_fit, img_per = S.local_storage_fit(b64, S.utf16_units("bg:bg-0001"))
    disk = 128 * 10 ** 9
    caps = {b: S.idb_cap_bytes(disk, b) for b in list(S.QUOTA_SHARE) + ["Firefox"]}
    return {
        "draft_chars_utf16": units, "draft_utf8_bytes": len(txt.encode()), "draft_key": key,
        "local_storage_limit_bytes": S.LOCAL_STORAGE_BYTES,
        "draft_bytes_in_local_storage": per, "drafts_that_fit": fit,
        "ai_png_bytes": png, "ai_png_base64_chars": b64, "ai_png_bytes_in_local_storage": img_per,
        "ai_pngs_that_fit": img_fit,
        "idb_example_disk_bytes": disk, "idb_caps_bytes": caps,
        "eviction": S.eviction_table(),
    }


def e10(p, a):
    ok = True
    for d in C.DESIGNS:
        per = C.usage_per_card(a, d)
        mon = C.monthly_usage(a, d)
        ok &= all(mon[k] == per[k] * mon["cards"] for k in per)
        c = C.vercel_cost(p, a, mon)
        ok &= c["usage"] == sum(c["parts"].values())
    # 손계산: 서버 미리보기 대기 = 40 + 0.48 + 30 + 32
    lt = C.latency_table(a)
    hand = F(40) + F(48, 100) + 30 + 32
    return {"per_card_times_cards_equals_month": ok, "preview_server_hand_ms": r(hand, 2),
            "preview_server_model_ms": r(lt["preview_server"], 2), "hand_equals_model": hand == lt["preview_server"]}


def derived(p, a, e5):
    """본문에 나오는 중간 계산값. 본문 숫자를 이 파일과 대조하려고 남긴다."""
    net, s = a["network"], a["server"]
    lt, lt_us = C.latency_table(a), C.latency_table(a, "iad1")
    a2 = json.loads(json.dumps(a))
    a2["traffic"]["users_per_month"] *= 2
    m2 = C.monthly_usage(a2, "decided")
    c2 = C.vercel_cost(p, a2, m2)
    caps = {k: v / 10 ** 9 for k, v in e9()["idb_caps_bytes"].items()}
    return {
        "preview_up_ms": r(C.transfer_ms(s["preview"]["request_bytes"], net["up_mbps"]), 3),
        "preview_down_ms": r(C.transfer_ms(s["preview"]["response_bytes"], net["down_mbps"]), 3),
        "export_down_ms": r(C.transfer_ms(s["export_png"]["response_bytes"], net["down_mbps"]), 3),
        "ai_wall_s": r(F(s["ai_background"]["cpu_ms"] + s["ai_background"]["wait_ms"], 1000), 3),
        "ai_memory_over_cpu": r(F(str(e4(p, a)["ai_background"]["memory_usd"])) / F(str(e4(p, a)["ai_background"]["active_cpu_usd"])), 2),
        "preview_iad1_minus_icn1_ms": r(lt_us["preview_server"] - lt["preview_server"], 2),
        "server_preview_bill_over_decided": r(F(str(e5["server_preview"]["bill_usd"])) / F(str(e5["decided"]["bill_usd"])), 2),
        "idb_caps_gb_128gb_disk": {k: r(v, 3) for k, v in caps.items()},
        "firefox_10pct_gb": r(F(128, 10), 1),
        "exercise_double_users": {"requests": m2["requests"], "usage_usd": r(c2["usage"], 4), "bill_usd": r(c2["bill"], 4),
                                  "cdn_tier_usd": c2["cdn_tier_usd"]},
    }


def main():
    p, a = C.load()
    times = C.keystroke_times(a["typing"])
    sent = C.debounced_requests(times, a["typing"]["debounce_ms"])
    e5, e6, e7 = e5_e6_e7(p, a)
    dec = C.monthly_usage(a, "decided")
    ai_krw = C.ai_model_cost_krw(a, dec)
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "checked": "2026-10-10", "prices": "data/prices_2026-10-10.json", "assumptions": "data/assumptions.json"},
        "e0_decisions": D.table(),
        "e1_requests_per_card": e1(a),
        "e2_latency_ms": {reg: {k: r(v, 3) for k, v in C.latency_table(a, reg).items()} for reg in ("icn1", "iad1")},
        "e3_debounce": {"keystrokes": len(times), "typing_ms": times[-1], "debounce_ms": a["typing"]["debounce_ms"],
                        "requests_without_debounce": len(times), "requests_with_debounce": len(sent),
                        "sent_at_ms": sent},
        "e4_one_request_icn1": e4(p, a),
        "e5_vercel_month": e5,
        "e6_vercel_hobby": e6,
        "e7_netlify_month": e7,
        "e8_ai_vs_hosting": {"ai_images": dec["cards"] * 2, "ai_krw": ai_krw,
                             "hosting_usage_usd_decided": e5["decided"]["usage_usd"],
                             "hosting_usage_krw_decided": r(F(str(e5["decided"]["usage_usd"])) * a["usd_krw"], 0),
                             "usd_krw_assumed": a["usd_krw"],
                             "png_on_server_extra_usage_usd": r(F(str(e5["decided"]["usage_usd"])) - F(str(e5["png_in_browser"]["usage_usd"])), 4),
                             "ai_one_call_krw": r(F(str(e4(p, a)["ai_background"]["total_usd"])) * a["usd_krw"], 4)},
        "e9_storage": e9(),
        "e10_crosscheck": e10(p, a),
        "e11_derived_for_text": derived(p, a, e5),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
