"""1인 개발 앱 출시 실전 5장: 무료 한도와 유료 전환 지점 계산.

가격·한도는 data/pricing_2026-10-10.json(공식 문서, 2026-10-10 확인), 사용 패턴은
data/usage_assumptions.json(가상)에서 읽어요. 금액은 달러, GB = 1,000 MB로 셉니다.

두 종류의 한도를 나눠요.
- 달마다 새로 세는 것: MAU, 내보내는 양(egress), Edge Function 호출, 정적 호스팅 전송량
- 쌓이는 것: DB 크기, 스토리지 크기. 활성 사용자 수가 그대로여도 달마다 늘어요.
스토리지 요금은 실제로는 GB-시간(평균)으로 매기지만, 여기서는 달 끝 크기로 계산해 조금 넉넉하게(비싸게) 잡아요.
내보내는 양은 캐시 여부를 나누지 않고 모두 캐시 안 된 쪽으로 셉니다(역시 보수적).
"""

import json
import math
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).parent
PRICING = json.loads((HERE / "data" / "pricing_2026-10-10.json").read_text(encoding="utf-8"))
USAGE = json.loads((HERE / "data" / "usage_assumptions.json").read_text(encoding="utf-8"))

F = Fraction


def per_user():
    """활성 사용자 1명의 한 달 사용량(MB, 회). 분수로 정확히 계산해요."""
    u = USAGE["per_active_user_month"]
    kb = USAGE["kb_per_mb"]
    return {
        "db_mb": F(u["new_cards"] * u["card_row_kb"], kb),
        "storage_mb": F(u["uploads"] * u["upload_kb"] + u["ai_backgrounds"] * u["ai_png_kb"], kb),
        "egress_mb": F(u["image_views"] * u["image_view_kb"], kb) + u["api_json_mb"],
        "edge_calls": u["edge_calls"],
        "static_mb": F(u["full_bundle_loads"] * u["bundle_kb"], kb),
        "web_requests": u["web_requests"],
        "ai_images": u["ai_backgrounds"],
    }


def free_walls():
    """Supabase 무료 한도 각각이 몇 명(달마다) 또는 몇 달(쌓이는 것)에서 넘치는지."""
    s = PRICING["supabase"]["free"]
    pu = per_user()
    gb = USAGE["mb_per_gb"]
    return {
        "mau": s["mau"],
        "egress_mau": F(s["egress_gb"] * gb) / pu["egress_mb"],
        "edge_mau": F(s["edge_invocations"]) / pu["edge_calls"],
        # 쌓이는 것: 사용자 수 x 달 수 가 이 값을 넘으면 넘쳐요
        "storage_user_months": F(s["storage_gb"] * gb) / pu["storage_mb"],
        "db_user_months": F(s["db_size_mb"]) / pu["db_mb"],
    }


def first_month_over(user_months_cap, mau):
    """활성 사용자 mau명이 매달 같은 양을 쌓을 때, 쌓인 양이 한도를 처음 넘는 달(1부터)."""
    return math.floor(F(user_months_cap) / mau) + 1


def free_status(mau, month):
    """mau명, month번째 달 끝에 Supabase 무료 한도 중 무엇이 넘쳤나."""
    s = PRICING["supabase"]["free"]
    pu = per_user()
    gb = USAGE["mb_per_gb"]
    used = {
        "mau": mau,
        "egress_gb": mau * pu["egress_mb"] / gb,
        "edge_calls": mau * pu["edge_calls"],
        "storage_gb": mau * pu["storage_mb"] * month / gb,
        "db_mb": mau * pu["db_mb"] * month,
    }
    over = []
    if used["mau"] > s["mau"]:
        over.append("MAU")
    if used["egress_gb"] > s["egress_gb"]:
        over.append("egress")
    if used["edge_calls"] > s["edge_invocations"]:
        over.append("Edge Function 호출")
    if used["storage_gb"] > s["storage_gb"]:
        over.append("스토리지")
    if used["db_mb"] > s["db_size_mb"]:
        over.append("DB 크기")
    return used, over


def pro_cost(mau, month, spend_cap_off=True):
    """Supabase Pro 한 달 요금(프로젝트 1개, Micro 컴퓨트). 스펜드 캡을 켜 두면 넘는 사용량은 막히고 25달러."""
    p = PRICING["supabase"]["pro"]
    o = p["overage"]
    used, _ = free_status(mau, month)
    lines = {"base": F(p["price_usd"]),
             "compute_after_credit": F(p["micro_compute_usd_month"]) - F(p["compute_credit_usd"])}
    over = {
        "mau": max(0, mau - p["mau"]),
        "storage_gb": max(F(0), used["storage_gb"] - p["storage_gb"]),
        "egress_gb": max(F(0), used["egress_gb"] - p["egress_gb"]),
        "edge_packs": math.ceil(max(0, used["edge_calls"] - p["edge_invocations"]) / 1_000_000),
        "db_gb": max(F(0), used["db_mb"] / USAGE["mb_per_gb"] - p["db_disk_gb"]),
    }
    if spend_cap_off:
        lines["mau"] = over["mau"] * F(str(o["mau_usd_each"]))
        lines["storage"] = over["storage_gb"] * F(str(o["storage_usd_per_gb_month"]))
        lines["egress"] = over["egress_gb"] * F(str(o["egress_usd_per_gb"]))
        lines["edge"] = over["edge_packs"] * o["edge_usd_per_million"]
        lines["db_disk"] = over["db_gb"] * F(str(o["db_disk_usd_per_gb"]))
    total = sum(lines.values())
    blocked = [k for k, v in over.items() if v > 0] if not spend_cap_off else []
    return {"total": total, "lines": lines, "over": over, "blocked_with_cap": blocked}


def vercel_hobby_mau():
    h = PRICING["vercel"]["hobby"]
    pu = per_user()
    return {
        "transfer_mau": F(h["fast_data_transfer_gb"] * USAGE["mb_per_gb"]) / pu["static_mb"],
        "requests_mau": F(h["cdn_requests"]) / pu["web_requests"],
    }


def netlify_free(mau, deploys=None):
    """Netlify 무료 300크레딧 중 몇 크레딧을 쓰나(배포 + 전송 + 요청)."""
    n = PRICING["netlify"]
    c = n["credit_costs"]
    pu = per_user()
    deploys = USAGE["deploys_per_month"] if deploys is None else deploys
    parts = {
        "deploys": F(deploys * c["production_deploy"]),
        "bandwidth": mau * pu["static_mb"] / USAGE["mb_per_gb"] * c["bandwidth_per_gb"],
        "requests": F(mau * pu["web_requests"], 10_000) * c["web_requests_per_10k"],
    }
    total = sum(parts.values())
    per_user_credit = (pu["static_mb"] / USAGE["mb_per_gb"] * c["bandwidth_per_gb"]
                       + F(pu["web_requests"], 10_000) * c["web_requests_per_10k"])
    max_mau = (n["free"]["credits"] - parts["deploys"]) / per_user_credit
    return {"parts": parts, "total": total, "within": total <= n["free"]["credits"],
            "per_user_credit": per_user_credit, "max_mau_at_deploys": max_mau}


def ai_cost(mau):
    return mau * per_user()["ai_images"] * F(str(USAGE["hypothetical_ai_cost_usd_per_image"]))


def to_float(x, nd=4):
    return round(float(x), nd)
