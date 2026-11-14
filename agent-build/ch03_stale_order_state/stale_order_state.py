"""3장: 에이전트의 조회 -> 판단 -> 실행 사이에 다른 작업자가 주문 상태를 바꾸는 모든 끼어들기 순서를 나열해 센다.

실제 스레드 경쟁은 재지 않아요. 두 작업자의 단계를 줄 세우는 모든 방법을 itertools로 나열하고,
각 순서를 한 번씩 차례로 실행해요. 난수와 시간이 없어서 몇 번을 돌려도 같아요.

주문 상태와 전이표(가상):
  결제됨  --ship-->            배송중
  결제됨  --cancel-->          취소됨   (고객이 직접 취소하면 결제 금액이 바로 환불돼요: 지급 1회)
  결제됨  --refund-->          환불됨   (에이전트의 환불: 지급 1회)
  배송중  --request_return-->  반품접수
  반품접수 --complete_return--> 환불됨   (반품 절차의 환불: 지급 1회)
환불 정책: "배송 전(결제됨)인 주문만 에이전트가 바로 환불한다."
다른 작업자의 행동은 늘 전이표의 시작 상태를 확인하고 실행해요(자기 시스템의 화면이 하는 검사).
"""
from itertools import combinations

TRANSITIONS = {
    ("결제됨", "ship"): ("배송중", 0),
    ("결제됨", "cancel"): ("취소됨", 1),
    ("결제됨", "refund"): ("환불됨", 1),
    ("배송중", "request_return"): ("반품접수", 0),
    ("반품접수", "complete_return"): ("환불됨", 1),
}

WORKERS = {
    "W1 배송 출고": ["ship"],
    "W2 고객 직접 취소": ["cancel"],
    "W3 출고 후 반품 접수": ["ship", "request_return"],
    "W4 출고 후 반품 접수와 완료": ["ship", "request_return", "complete_return"],
}

VARIANTS = {
    "V0": "조회 결과만 믿고 실행(실행 때 검사 없음)",
    "V1": "실행할 때 전이표로 검사(결제됨일 때만 환불)",
    "V2": "실행 직전에 한 번 더 조회(실행 때 검사는 없음)",
}


def agent_steps(variant):
    return ["lookup", "decide", "recheck", "execute"] if variant == "V2" else ["lookup", "decide", "execute"]


def interleavings(worker_len, agent_len):
    """두 줄의 단계를 각자 순서를 지키며 섞는 모든 방법. 작업자 단계가 들어갈 칸을 고른다."""
    total = worker_len + agent_len
    for worker_slots in combinations(range(total), worker_len):
        yield ["W" if i in worker_slots else "A" for i in range(total)]


def run_one(order, variant, worker_actions):
    state = "결제됨"
    payouts = 0
    seen = None
    plan = False
    agent = iter(agent_steps(variant))
    worker = iter(worker_actions)
    result = {"attempted": False, "executed": False, "stale": False, "blocked": False, "unsafe": False}
    for who in order:
        if who == "W":
            action = next(worker)
            move = TRANSITIONS.get((state, action))
            if move:  # 작업자는 자기 검사를 통과할 때만 실행돼요
                state, pay = move
                payouts += pay
            continue
        step = next(agent)
        if step == "lookup":
            seen = state
        elif step == "decide":
            plan = seen == "결제됨"  # 배송 전이면 환불하기로 정해요
        elif step == "recheck":
            plan = plan and state == "결제됨"
        elif step == "execute" and plan:
            result["attempted"] = True
            result["stale"] = state != seen
            if variant == "V1" and (state, "refund") not in TRANSITIONS:
                result["blocked"] = True  # 전이표가 거절: 돈이 나가지 않아요
            else:
                result["executed"] = True
                result["unsafe"] = state != "결제됨"  # 결제됨이 아닌 상태(출고·반품 중·취소됨·환불됨)에서 지급
                state = "환불됨"
                payouts += 1
    result["payouts"] = payouts
    result["double"] = payouts >= 2
    result["final_state"] = state
    return result


def run_all(variant):
    per_worker = {}
    for name, actions in WORKERS.items():
        rows = [run_one(o, variant, actions) for o in interleavings(len(actions), len(agent_steps(variant)))]
        per_worker[name] = {
            "orders": len(rows),
            "attempted": sum(r["attempted"] for r in rows),
            "stale_attempts": sum(r["stale"] for r in rows),
            "executed": sum(r["executed"] for r in rows),
            "blocked": sum(r["blocked"] for r in rows),
            "double_payout": sum(r["double"] for r in rows),
            "unsafe_refund": sum(r["unsafe"] for r in rows),
        }
    total = {k: sum(w[k] for w in per_worker.values()) for k in next(iter(per_worker.values()))}
    return {"per_worker": per_worker, "total": total}


if __name__ == "__main__":
    for v, label in VARIANTS.items():
        res = run_all(v)
        print(v, label)
        for name, w in res["per_worker"].items():
            print("  ", name, w)
        print("   합계", res["total"])
