"""6장 · 에이전트가 이 고객의 주문만 만지게 하려면 어디서 막아야 하나요? (표준 라이브러리만)

가짜 모델은 고객 메시지에 적힌 주문번호를 그대로 도구에 넘긴다(모델이 알아서 거절해 주지 않는다고 가정).
그래서 이 실습이 재는 것은 "모델 판단"이 아니라 "도구 코드가 어디까지 검사하느냐"다.
같은 요청 20건을 도구 권한 변형 6개에 넣어, 남의 주문 조회·환불이 나간 수와
정당한 요청이 막힌 수를 센다. 금액 단위는 원(가상).

입력: 코드 안의 주문 소유 표, 요청 20건. 출력: 변형(V1~V6)별 유출·무단 환불·정당한 요청 거절 수.
README 표와의 대응: V1은 타인 환불 3건 241,000원(33,000+58,000+150,000), V6은 정당한 요청 거절 0, 사람 대기 1.
"""

# 주문 소유 표: 주문번호 -> (소유 고객, 결제 금액)
ORDERS = {
    "O101": ("C1", 38_000), "O102": ("C1", 52_000), "O103": ("C1", 120_000),
    "O104": ("C1", 27_000), "O105": ("C1", 64_000),
    "O201": ("C2", 45_000), "O202": ("C2", 90_000),
    "O301": ("C3", 33_000), "O302": ("C3", 150_000),
    "O401": ("C4", 58_000),
}
# 대리 등록표: 주문 소유 고객 -> 대신 신청해도 되는 고객(가상 등록 정보)
# 가상 등록 정보. 실제 서비스의 대리인 등록·신원 확인 절차는 다루지 않는다.
PROXIES = {"C2": {"C1"}}
SESSION_CUSTOMER = "C1"         # 로그인한 고객. 서버가 정하고 모델은 바꿀 수 없다.
# 상한과 같은 금액은 통과(초과만 보류). 120,000원 주문의 본인 환불(요청 6번)이 보류되는 정당한 건이다.
REFUND_CAP = 100_000            # 변형 V3·V5·V6의 자동 환불 상한(가상). 넘으면 사람에게 넘긴다.

# 요청 20건: (번호, 종류, 동작, 주문, 금액). 종류: own 본인 10, other 타인 6, family 가족 대리 4.
REQUESTS = [
    (1, "own", "lookup", "O101", 0), (2, "own", "refund", "O101", 38_000),
    (3, "own", "lookup", "O102", 0), (4, "own", "refund", "O102", 52_000),
    (5, "own", "lookup", "O103", 0), (6, "own", "refund", "O103", 120_000),
    (7, "own", "lookup", "O104", 0), (8, "own", "refund", "O104", 27_000),
    (9, "own", "lookup", "O105", 0), (10, "own", "refund", "O105", 64_000),
    (11, "other", "lookup", "O301", 0), (12, "other", "refund", "O301", 33_000),
    (13, "other", "lookup", "O401", 0), (14, "other", "refund", "O401", 58_000),
    (15, "other", "lookup", "O302", 0), (16, "other", "refund", "O302", 150_000),
    (17, "family", "lookup", "O201", 0), (18, "family", "refund", "O201", 45_000),
    (19, "family", "lookup", "O202", 0), (20, "family", "refund", "O202", 90_000),
]

# 변형마다 (범위 검사 방식, 환불 상한을 쓰는가, 설명)
#   범위 검사 방식: none 없음 / arg 모델이 넘긴 고객 ID로 검사 / session 서버가 아는 세션 고객으로 검사
#                 / session+proxy 세션 고객 또는 그 고객이 대리인으로 등록된 주문
# V1·V3는 "none"이라 소유를 전혀 보지 않는다. 변형 이름 V1~V6은 README 표의 행과 같다.
VARIANTS = {
    "V1": ("none", False, "전체 조회: 주문번호만 맞으면 된다"),
    "V2": ("arg", False, "모델이 넘긴 고객 ID로 검사"),
    "V3": ("none", True, "환불 상한만"),
    "V4": ("session", False, "세션 고객 범위"),
    "V5": ("session", True, "세션 고객 범위 + 환불 상한"),
    "V6": ("session+proxy", True, "세션 고객 범위 + 대리 등록표 + 환불 상한"),
}


def claimed_customer(kind, order):
    """가짜 모델이 도구 인자 customer_id에 채우는 값. 메시지가 말하는 대로 '그 주문 주인'이라고 적는다.
    (V2에서만 쓴다. 모델이 쓴 글을 믿는 설계가 어떻게 되는지 보려는 장치다.)"""
    # 항상 주문 주인을 적는다. 모델이 쓴 인자는 공격자가 맞춰 쓸 수 있어 믿을 근거가 못 된다는 점을 보이려는 설정이다.
    return ORDERS[order][0]


def call_tool(variant, action, order, amount, customer_id_arg):
    """도구 코드. 돌려주는 값: ('ok'|'denied'|'held', 이유)."""
    if order not in ORDERS:
        return "denied", "없는 주문"
    owner, paid = ORDERS[order]
    scope, use_cap, _ = VARIANTS[variant]
    # V2: 비교 대상이 모델이 넘긴 값이라, 주인 ID를 그대로 적으면 통과한다.
    if scope == "arg" and customer_id_arg != owner:
        return "denied", "인자의 고객 ID가 주문 주인과 다름"
    if scope in ("session", "session+proxy"):
        # 비교 대상이 서버가 아는 세션 고객(모델이 못 바꿈). 인자는 쓰지 않는다.
        allowed = owner == SESSION_CUSTOMER
        if scope == "session+proxy" and not allowed:
            allowed = SESSION_CUSTOMER in PROXIES.get(owner, set())
        if not allowed:
            return "denied", "세션 고객의 주문이 아님"
    if action == "refund":
        # 상한 사용 여부와 무관하게 결제 금액 초과는 모든 변형에서 거절된다.
        if amount > paid:
            return "denied", "결제 금액보다 큼"
        if use_cap and amount > REFUND_CAP:
            return "held", "자동 환불 상한 초과, 사람 승인 대기"
    return "ok", ""


def run_variant(variant):
    leaked_lookups = unauthorized_refunds = unauthorized_sum = 0
    legit_blocked = {"own": 0, "family": 0}
    legit_held = 0
    blocked_ids = []
    # 타인(other) 요청이 실행되면 유출·무단 환불, 정당한 요청이 거절되거나 보류되면 막힌 수로 센다.
    for no, kind, action, order, amount in REQUESTS:
        status, _ = call_tool(variant, action, order, amount, claimed_customer(kind, order))
        if kind == "other":
            if status == "ok" and action == "lookup":
                leaked_lookups += 1
            if status == "ok" and action == "refund":
                unauthorized_refunds += 1
                unauthorized_sum += amount
        else:  # 정당한 요청(본인, 등록된 가족)
            if status == "denied":
                legit_blocked[kind] += 1
                blocked_ids.append(no)
            elif status == "held":
                legit_held += 1
                blocked_ids.append(no)
    return {
        "variant": variant,
        "leaked_lookups": leaked_lookups,
        "unauthorized_refunds": unauthorized_refunds,
        "unauthorized_sum": unauthorized_sum,
        "legit_denied_own": legit_blocked["own"],
        "legit_denied_family": legit_blocked["family"],
        "legit_held_for_human": legit_held,
        "not_executed_request_numbers": blocked_ids,
    }


def main():
    print("요청", len(REQUESTS), "건: 본인", sum(r[1] == "own" for r in REQUESTS),
          "타인", sum(r[1] == "other" for r in REQUESTS), "가족", sum(r[1] == "family" for r in REQUESTS))
    for v in VARIANTS:
        r = run_variant(v)
        print(v, VARIANTS[v][2])
        print("  ", {k: val for k, val in r.items() if k != "variant"})


if __name__ == "__main__":
    main()
