"""8장 · 대화와 도구 결과가 쌓이면 에이전트는 환불 정책의 어느 조항을 잃나요? (표준 라이브러리만)

컨텍스트(모델이 한 번에 읽는 글 전체)에 예산을 두고, 예산을 넘으면 어떤 글이 먼저 잘려 나가는지 센다.
토큰 대신 글자 수를 대용으로 쓴다(실제 토큰 수와 다르다). 정책 조항 8개의 길이는 아래 문장의 실제 글자 수이고,
대화·도구 결과의 길이는 턴 번호로 정한 가상 값이다. 가짜 에이전트는 조항이 컨텍스트에 있으면 거절·사람에게 넘기기로,
없으면 환불 승인으로 답하는 규칙표다. 실제 모델이 긴 글에서 조항을 얼마나 놓치는지는 재지 않는다.
"""

BUDGET = 6_000   # 글자 수 예산(가상). 실제 모델의 창 크기가 아니다.

HEADER = "당신은 환불 문의를 처리하는 도우미입니다. 도구 결과와 정책에 근거해서만 답하세요."
POLICY = [
    ("P1", "환불 불가 품목", "환불 불가 품목: 속옷·양말 같은 위생용품은 포장을 뜯은 뒤에는 환불할 수 없다. 이 경우 환불을 실행하지 말고 거절하며, 거절하는 이유와 교환이 가능한 경우를 고객에게 안내한다. 불량이 확인된 경우는 사람 상담원에게 넘긴다."),
    ("P2", "환불 기한", "환불 기한: 수령 후 7일 안의 주문만 자동으로 환불할 수 있다. 7일이 지난 주문은 자동 환불하지 않고 사람 상담원에게 넘기며, 넘길 때 주문번호와 수령일, 고객이 적은 사유를 함께 전달한다."),
    ("P3", "승인 문턱", "승인 문턱: 한 건의 환불 금액이 50,000원을 넘으면 사람 승인을 받은 뒤에만 실행한다. 같은 주문의 환불을 여러 번에 나눠 요청해도 그 주문의 누적 금액으로 계산하며, 승인이 나기 전에는 고객에게 환불이 끝났다고 말하지 않는다."),
    ("P4", "부분 환불", "부분 환불: 같은 주문의 누적 환불 금액이 결제 금액을 넘으면 실행하지 않는다. 일부 상품만 돌려보내는 경우에는 돌려보낸 상품의 결제 금액만큼만 환불하고, 배송비는 상품 불량일 때만 함께 환불한다."),
    ("P5", "배송 중 주문", "배송 중 주문: 아직 배송 중이거나 반송 접수 전인 주문은 환불하지 않는다. 대신 반송 접수 방법을 안내하고, 반송 상품이 창고에 도착한 것이 기록으로 확인된 뒤에 환불 절차를 시작한다."),
    ("P6", "쿠폰·포인트", "쿠폰·포인트: 쿠폰으로 할인받은 금액은 현금이 아니라 같은 종류의 쿠폰으로 되돌린다. 포인트로 결제한 금액은 포인트로 되돌리고, 현금 결제 금액만 결제 수단으로 환불한다."),
    ("P7", "대리 신청", "대리 신청: 주문한 본인이 아니면 등록된 대리인만 환불을 신청할 수 있다. 대리인 등록이 확인되지 않으면 조회도 하지 말고 사람 상담원에게 넘긴다. 가족이라고 말하는 것만으로는 대리인으로 보지 않는다."),
    ("P8", "기록 남기기", "기록 남기기: 환불을 실행하거나 거절하거나 사람에게 넘길 때마다 요청 번호, 처리한 단계, 이유를 기록 도구에 남긴다. 기록 도구 호출이 실패하면 환불 실행을 멈추고 사람에게 알린다."),
]


def turn_items(i):
    """i번째 턴(1부터)의 글 세 덩어리: 고객 메시지, 도구 결과, 에이전트 답. 길이는 턴 번호로 정한 가상 값."""
    return [
        ("user", f"T{i}", 90 + 10 * (i % 4)),
        ("tool", f"T{i}", 420 + 70 * (i % 3)),
        ("assistant", f"T{i}", 150 + 15 * (i % 3)),
    ]


def build_context(n_turns):
    """컨텍스트 항목 목록. 항목은 (종류, 이름, 글자 수, 고정 여부)이고 앞쪽이 가장 오래된 글이다."""
    items = [("system", "header", len(HEADER), True)]
    items += [("policy", pid, len(text), True) for pid, _, text in POLICY]
    for i in range(1, n_turns + 1):
        items += [(k, name, size, False) for k, name, size in turn_items(i)]
    return items


def total(items):
    return sum(i[2] for i in items)


PLACEHOLDER = 40   # 지운 도구 결과 자리에 남기는 글자 수(예: "[도구 결과 삭제됨]" 같은 표시)
KEEP_RECENT_TURNS = 2


def apply_strategy(items, strategy, budget=BUDGET):
    """S1 앞에서부터 자르기 / S2 정책 고정 + 오래된 대화부터 자르기 / S3 정책 고정 + (예산을 넘으면) 최근 2턴 밖의 도구 결과를 지운 뒤 그래도 넘으면 자르기."""
    items = list(items)
    if strategy == "S3" and total(items) > budget:     # 예산을 넘을 때만 오래된 도구 결과를 지운다
        last_turns = {f"T{t}" for t in _turn_numbers(items)[-KEEP_RECENT_TURNS:]}
        items = [(k, n, PLACEHOLDER, p) if (k == "tool" and n not in last_turns) else (k, n, s, p)
                 for k, n, s, p in items]
    while total(items) > budget:
        for idx, it in enumerate(items):
            if strategy == "S1" or not it[3]:      # S1은 고정 개념이 없다. S2·S3은 고정 항목을 건너뛴다.
                del items[idx]
                break
        else:
            raise ValueError("고정한 항목만으로 예산을 넘었어요")
    return items


def _turn_numbers(items):
    seen = []
    for k, n, _s, _p in items:
        if k in ("user", "tool", "assistant") and int(n[1:]) not in seen:
            seen.append(int(n[1:]))
    return seen


def lost_clauses(items):
    present = {n for k, n, _s, _p in items if k == "policy"}
    return [pid for pid, _, _ in POLICY if pid not in present]


def fake_agent_decisions(items):
    """조항 Pk가 필요한 요청 8개(조항마다 하나). 조항이 있으면 올바른 처리, 없으면 환불 승인."""
    present = {n for k, n, _s, _p in items if k == "policy"}
    return {pid: ("올바름" if pid in present else "환불 승인(틀림)") for pid, _, _ in POLICY}


def run(n_turns, strategy, budget=BUDGET):
    before = build_context(n_turns)
    after = apply_strategy(before, strategy, budget)
    lost = lost_clauses(after)
    kept_turn1_user = any(k == "user" and n == "T1" for k, n, _s, _p in after)
    kept_user_messages = sum(1 for k, _n, _s, _p in after if k == "user")
    kept_tool_results = sum(1 for k, _n, s, _p in after if k == "tool" and s > PLACEHOLDER)
    return {
        "n_turns": n_turns, "strategy": strategy,
        "chars_before": total(before), "chars_after": total(after),
        "overflow_chars": max(0, total(before) - budget),
        "lost_header": not any(k == "system" for k, _n, _s, _p in after),
        "dropped_user_messages": n_turns - kept_user_messages,
        "lost_clauses": lost, "wrong_decisions": len(lost),
        "kept_turn1_user_message": kept_turn1_user,
        "kept_full_tool_results": kept_tool_results,
    }


def first_overflow_turn(budget=BUDGET):
    n = 0
    while total(build_context(n)) <= budget:
        n += 1
    return n


def main():
    pol = sum(len(t) for _, _, t in POLICY)
    print("정책 8개 글자 수", [len(t) for _, _, t in POLICY], "합계", pol, "머리말", len(HEADER), "예산", BUDGET)
    print("턴별 글자 수", [sum(s for _, _, s in turn_items(i)) for i in range(1, 6)], "...")
    print("예산을 처음 넘는 턴", first_overflow_turn())
    for strategy in ("S1", "S2", "S3"):
        print(strategy)
        for n in (6, 7, 8, 10, 12, 15, 20, 30):
            r = run(n, strategy)
            print("  ", r["n_turns"], r["chars_before"], r["overflow_chars"], r["chars_after"], "머리말 잃음" if r["lost_header"] else "머리말 유지",
                  r["lost_clauses"], r["wrong_decisions"], "잘린 고객 메시지", r["dropped_user_messages"],
                  "전체 도구 결과", r["kept_full_tool_results"])


if __name__ == "__main__":
    main()
