"""5장 · 금액이 얼마부터 사람에게 물어야 하나요? (표준 라이브러리만, 난수 없음)

환불 요청 100건(refund_requests.json, 가상 금액)에 승인 문턱을 바꿔 가며
  - 사람 승인이 필요한 건수
  - 승인 없이 나가는 건수·합계·최대 금액
  - 승인 때문에 늘어나는 사람 시간과 고객 대기(가상 값)
를 센다. 이어서 같은 주문의 환불을 여러 요청으로 쪼갰을 때 문턱이 뚫리는지,
주문 누적 합계로 검사하면 막히는지 센다.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMOUNTS = json.loads((HERE / "refund_requests.json").read_text(encoding="utf-8"))["amounts"]

# 가상 가정. 실제 승인자의 처리 시간이 아니다.
MINUTES_PER_APPROVAL = 3     # 승인자 1명이 한 건을 보는 데 드는 시간(분)
WAIT_PER_APPROVAL = 30       # 승인 대기열에서 고객이 기다리는 시간(분)

THRESHOLDS = [10_000, 20_000, 30_000, 50_000, 80_000, 100_000, 150_000, 250_000]


def needs_approval(amount, threshold):
    """요청 한 건의 금액이 문턱을 넘으면 사람 승인이 필요하다. 코드가 정한다(모델 판단 아님)."""
    return amount > threshold


def sweep(amounts, threshold):
    auto = [a for a in amounts if not needs_approval(a, threshold)]
    held = [a for a in amounts if needs_approval(a, threshold)]
    return {
        "threshold": threshold,
        "approvals": len(held),
        "auto_count": len(auto),
        "auto_sum": sum(auto),
        "auto_max": max(auto) if auto else 0,
        "held_sum": sum(held),
        "held_money_share_pct": round(sum(held) / sum(amounts) * 100, 1),
        "approver_minutes": len(held) * MINUTES_PER_APPROVAL,
        "customer_wait_minutes": len(held) * WAIT_PER_APPROVAL,
    }


# 쪼개기 시나리오(고정 목록): 주문 -> 그 주문에 대해 순서대로 들어온 환불 요청 금액.
# O5는 쪼개지 않은 대조군이다.
SPLIT_REQUESTS = [
    ("O1", [40_000, 40_000, 40_000]),
    ("O2", [45_000, 45_000]),
    ("O3", [50_000, 50_000, 50_000]),
    ("O4", [30_000, 30_000]),
    ("O5", [49_000]),
    ("O6", [49_000, 49_000, 49_000, 49_000, 4_000]),
]


def run_split(policy, threshold):
    """policy: 'per_request'(요청 한 건씩 검사) 또는 'per_order'(그 주문의 누적 합계로 검사).

    승인이 필요한 요청은 사람에게 가고(held), 나머지는 바로 나간다(auto).
    사람이 승인하기 전에는 누적 합계에 넣지 않는다(보류된 금액은 나가지 않았으므로).
    """
    auto_paid = {}
    held_count = 0
    for order, requests in SPLIT_REQUESTS:
        paid = 0
        for amount in requests:
            checked = amount if policy == "per_request" else paid + amount
            if needs_approval(checked, threshold):
                held_count += 1
            else:
                paid += amount
        auto_paid[order] = paid
    return {
        "policy": policy,
        "auto_paid_by_order": auto_paid,
        "auto_total": sum(auto_paid.values()),
        "auto_over_threshold_orders": sorted(o for o, p in auto_paid.items() if p > threshold),
        "max_auto_per_order": max(auto_paid.values()),
        "held_requests": held_count,
    }


def main():
    print("요청 수", len(AMOUNTS), "합계", sum(AMOUNTS), "최대", max(AMOUNTS))
    print("threshold approvals auto_count auto_sum auto_max approver_min customer_wait_min held_money_share_pct")
    for t in THRESHOLDS:
        r = sweep(AMOUNTS, t)
        print(t, r["approvals"], r["auto_count"], r["auto_sum"], r["auto_max"],
              r["approver_minutes"], r["customer_wait_minutes"], r["held_money_share_pct"])
    print("-- 쪼개기, 문턱 50000")
    total_requested = sum(sum(v) for _, v in SPLIT_REQUESTS)
    print("요청 합계", total_requested)
    for policy in ("per_request", "per_order"):
        print(run_split(policy, 50_000))


if __name__ == "__main__":
    main()
