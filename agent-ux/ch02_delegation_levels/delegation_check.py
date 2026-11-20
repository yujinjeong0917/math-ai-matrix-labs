"""2장 실습: 가상 환불 에이전트의 단계별 맡김 수준을 점검표로 확인한다.

자동화 유형 네 가지(정보 획득, 정보 분석, 결정, 행동)는 Parasuraman 외(2000)의 구분을 질문 틀로 옮긴 것이다.
맡김 수준 세 가지와 점검 규칙 두 개는 이 장이 만든 단순화이고 논문의 기준이 아니다.
모든 설계안은 가상, 예시용이다. 표준 라이브러리만 쓴다.
"""

LEVELS = ["보여 주기만", "제안하고 사람이 확정", "알아서 실행"]
TYPES = ["정보 획득", "정보 분석", "결정", "행동"]
AUTO = "알아서 실행"

# (단계, 유형, 되돌릴 수 없는가). 되돌릴 수 없는 분류는 가상 환불 단계표의 가정이다(환불 지급과 이미 보낸 안내).
STEPS = [
    ("주문 조회", "정보 획득", False),
    ("규정 대조와 금액 계산", "정보 분석", False),
    ("환불 승인 또는 거절 고르기", "결정", False),
    ("환불 지급", "행동", True),
    ("고객 안내 전송", "행동", True),
    ("상담 기록", "행동", False),
]

# 설계안: 단계별 (맡김 수준, 사용자 화면에 과정이 보이는가)
DESIGNS = {
    "A 모두 맡기고 요약만 보기": [
        (AUTO, True), (AUTO, True), (AUTO, False), (AUTO, False), (AUTO, False), (AUTO, False),
    ],
    "B 결정과 지급과 안내는 사람이 확정": [
        (AUTO, True), (AUTO, True), ("제안하고 사람이 확정", True),
        ("제안하고 사람이 확정", True), ("제안하고 사람이 확정", True), (AUTO, True),
    ],
    "C 결정은 맡기고 지급은 확정": [
        (AUTO, True), (AUTO, True), (AUTO, True),
        ("제안하고 사람이 확정", True), (AUTO, True), (AUTO, True),
    ],
}


def warnings(design):
    """점검 규칙 두 개. W1: 되돌릴 수 없는 단계를 사람 확인 없이 실행, W2: 결정과 행동을 알아서 하는데 과정이 안 보임."""
    out = []
    for (name, typ, irreversible), (level, visible) in zip(STEPS, design):
        if irreversible and level == AUTO:
            out.append(("W1", name))
        if typ in ("결정", "행동") and level == AUTO and not visible:
            out.append(("W2", name))
    return out


def counts(design):
    w = warnings(design)
    return {"W1": sum(1 for r, _ in w if r == "W1"), "W2": sum(1 for r, _ in w if r == "W2"), "합": len(w)}


def changed_steps(a, b):
    return [STEPS[i][0] for i, (x, y) in enumerate(zip(a, b)) if x != y]


def main():
    print("[1] 단계와 자동화 유형")
    for name, typ, irr in STEPS:
        print("  %s | %s | 되돌릴 수 %s" % (name, typ, "없음" if irr else "있음"))
    print("[2] 설계안별 경고(W1 되돌릴 수 없는 단계를 확인 없이 실행, W2 결정과 행동을 안 보이게 맡김)")
    for name, d in DESIGNS.items():
        c = counts(d)
        print("  %s: W1 %d개, W2 %d개, 합 %d개" % (name, c["W1"], c["W2"], c["합"]))
    print("[3] A의 경고 단계")
    for rule, name in warnings(DESIGNS["A 모두 맡기고 요약만 보기"]):
        print("  %s %s" % (rule, name))
    keys = list(DESIGNS)
    print("[4] A에서 B로 바꾼 단계 %d개: %s" % (
        len(changed_steps(DESIGNS[keys[0]], DESIGNS[keys[1]])), ", ".join(changed_steps(DESIGNS[keys[0]], DESIGNS[keys[1]]))))
    print("[5] 단계 %d개를 유형 %d개와 수준 %d개로 나누면 칸 %d개" % (len(STEPS), len(TYPES), len(LEVELS), len(TYPES) * len(LEVELS)))


if __name__ == "__main__":
    main()
