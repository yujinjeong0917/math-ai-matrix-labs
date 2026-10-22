"""환불이 두 번 나가는지 변형 A~D로 세어 results/ch04.json에 저장해요.
난수가 없어서 몇 번을 돌려도 같은 값이 나와요. 웹 챕터의 표는 이 파일의 숫자를 그대로 옮겼어요.
실행: python3 experiments.py"""
import json
from pathlib import Path

from double_refund import (DUPLICATE_MESSAGE_ORDERS, LATE_REPLY_ORDERS, refund_amounts,
                           run_variant, simulate)

OUT = Path(__file__).resolve().parent / "results" / "ch04.json"
VARIANT_NAMES = {"A": "방어 없음", "B": "보내기 전에 상태 조회", "C": "주문마다 같은 번호표(멱등성 키)", "D": "시도마다 새 번호표"}


def main():
    planted = sorted(LATE_REPLY_ORDERS | DUPLICATE_MESSAGE_ORDERS)
    variants = {}
    for v in "ABCD":
        refunds, total, extra = run_variant(v)
        payouts = simulate(v).payouts
        doubled = sorted({o for o, _ in payouts if sum(1 for p, _ in payouts if p == o) > 1})
        variants[v] = {"name": VARIANT_NAMES[v], "refund_count": refunds, "total_paid": total,
                       "extra_over_correct": extra, "blocked_of_planted": len(planted) - len(doubled),
                       "orders_paid_twice": doubled}
    res = {"orders": len(refund_amounts), "correct_total": sum(refund_amounts.values()),
           "planted_failure_orders": planted, "late_reply_orders": sorted(LATE_REPLY_ORDERS),
           "duplicate_message_orders": sorted(DUPLICATE_MESSAGE_ORDERS), "variants": variants,
           "note": "요청이 차례로 오는 장난감 실험이에요. 동시 요청, 키 만료, 같은 키에 다른 내용은 재지 않았어요."}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
