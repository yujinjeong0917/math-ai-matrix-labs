"""5장 · 금액이 얼마부터 사람에게 물어야 하나요? (표준 라이브러리만, 난수 없음)

환불 요청 100건(refund_requests.json, 가상 금액)에 승인 문턱을 바꿔 가며
  - 사람 승인이 필요한 건수
  - 승인 없이 나가는 건수·합계·최대 금액
  - 승인 때문에 늘어나는 사람 시간과 고객 대기(가상 값)
를 센다. 이어서 같은 주문의 환불을 여러 요청으로 쪼갰을 때 문턱이 뚫리는지,
주문 누적 합계로 검사하면 막히는지 센다.

입력: refund_requests.json의 금액 100개(원), 코드 안의 쪼개기 시나리오 6개 주문.
출력: 문턱별 승인 건수·자동 지급 합계·최대, 쪼개기 정책별 자동 지급 합계.
README 표와의 대응: 문턱 50,000원이면 승인 31건, 자동 합계 1,652,000원, 승인자 시간 93분(=31x3).
쪼개기: 요청별 검사 669,000원이 그대로 나가고, 주문 누적 검사는 263,000원만 나가며 10건이 보류된다.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
AMOUNTS = json.loads((HERE / "refund_requests.json").read_text(encoding="utf-8"))["amounts"]

# 가상 가정. 실제 승인자의 처리 시간이 아니다.
# 승인 시간(분)은 이 장이 가정한 값. 실제 승인자의 처리 시간을 잰 것이 아니다.
MINUTES_PER_APPROVAL = 3     # 승인자 1명이 한 건을 보는 데 드는 시간(분)
WAIT_PER_APPROVAL = 30       # 승인 대기열에서 고객이 기다리는 시간(분)

THRESHOLDS = [10_000, 20_000, 30_000, 50_000, 80_000, 100_000, 150_000, 250_000]


def needs_approval(amount, threshold):
    """요청 한 건의 금액이 문턱을 넘으면 사람 승인이 필요하다. 코드가 정한다(모델 판단 아님)."""
    # 문턱과 같은 금액은 승인 없이 나간다(경계는 "초과"). 테스트가 50,000과 50,001로 확인한다.
    return amount > threshold


def sweep(amounts, threshold):
    """문턱 하나에 대해 승인 건수, 자동 지급 합계·최대, 승인자 시간, 고객 대기 시간을 센다."""
    auto = [a for a in amounts if not needs_approval(a, threshold)]
    held = [a for a in amounts if needs_approval(a, threshold)]
    return {
        "threshold": threshold,
        "approvals": len(held),
        "auto_count": len(auto),
        "auto_sum": sum(auto),
        # 자동 지급이 하나도 없으면 빈 목록의 max가 오류를 내므로 0으로 둔다.
        "auto_max": max(auto) if auto else 0,
        "held_sum": sum(held),
        # 보류된 금액이 전체 금액에서 차지하는 비율(%, 소수 첫째 자리 반올림)
        "held_money_share_pct": round(sum(held) / sum(amounts) * 100, 1),
        # 사람 시간과 고객 대기는 건수에 가정 값을 곱한 것이라 건수가 같으면 비례한다.
        "approver_minutes": len(held) * MINUTES_PER_APPROVAL,
        "customer_wait_minutes": len(held) * WAIT_PER_APPROVAL,
    }


# 쪼개기 시나리오(고정 목록): 주문 -> 그 주문에 대해 순서대로 들어온 환불 요청 금액.
# O5는 쪼개지 않은 대조군이다.
# 주문당 요청이 순서대로 들어온 목록. 한 요청은 문턱 이하여도 합치면 문턱을 넘는다(요청 쪼개기).
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
            # 검사하는 값만 다르다. 요청별은 이번 금액, 주문별은 이미 나간 금액 + 이번 금액.
            checked = amount if policy == "per_request" else paid + amount
            if needs_approval(checked, threshold):
                held_count += 1
            else:
                # 보류된 요청은 누적에 넣지 않는다(실제로 나가지 않았으므로). 이 가정에 따라 보류 건수가 달라질 수 있다.
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
