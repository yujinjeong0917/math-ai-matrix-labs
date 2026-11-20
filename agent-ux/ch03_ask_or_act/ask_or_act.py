"""3장 실습: Horvitz(1999)의 기대 효용 틀로 "가만히 있기, 묻기, 실행하기"의 경계를 계산한다.

기호는 논문과 같다. G는 이 환불이 옳은 처리라는 목표, A는 실행, D는 사용자에게 묻기, nA는 아무것도 안 하기다.
u(행동, 목표)는 0~1 효용이다. 모든 효용은 가상, 예시용이다.
논문은 임계 확률 p*를 닫힌 식으로 적지 않는다. 아래 p_star와 두 경계 식은 논문의 기대 효용 식에서
이 장이 직접 유도한 값이다. 에이전트가 내놓는 p가 보정된 확률이라는 보장은 논문에도 이 코드에도 없다.
표준 라이브러리만 쓴다.
"""

# 가상 효용 (옳은 환불이면 G, 아니면 nG)
U = {
    "AG": 1.0,    # 옳은 환불을 실행
    "AnG": 0.0,   # 틀린 환불을 실행(돈이 나가 되돌릴 수 없다고 가정)
    "nAG": 0.6,   # 옳은 환불을 하지 않음(사람이 직접 해야 해서 늦어짐)
    "nAnG": 1.0,  # 틀린 환불을 하지 않음
    "DG": 0.8,    # 옳은 환불을 묻고 진행(묻는 번거로움)
    "DnG": 0.7,   # 틀린 환불을 물어서 거름
}

EPS = 1e-9


def eu_act(p, u=U):
    return p * u["AG"] + (1 - p) * u["AnG"]


def eu_none(p, u=U):
    return p * u["nAG"] + (1 - p) * u["nAnG"]


def eu_dialog(p, u=U):
    return p * u["DG"] + (1 - p) * u["DnG"]


def p_star(u=U):
    """행동과 무행동의 기대 효용이 같아지는 p."""
    a = u["nAnG"] - u["AnG"]
    b = u["AG"] - u["nAG"]
    return a / (a + b)


def p_none_dialog(u=U):
    a = u["nAnG"] - u["DnG"]
    b = u["DG"] - u["nAG"]
    return a / (a + b)


def p_dialog_act(u=U):
    a = u["DnG"] - u["AnG"]
    b = u["AG"] - u["DG"]
    return a / (a + b)


def decide(p, u=U):
    """기대 효용이 가장 큰 선택. 동점이면 더 조심스러운 쪽(가만히, 묻기, 실행 순서)."""
    options = [("가만히", eu_none(p, u)), ("묻기", eu_dialog(p, u)), ("실행", eu_act(p, u))]
    best = options[0]
    for name, v in options[1:]:
        if v > best[1] + EPS:
            best = (name, v)
    return best[0]


def main():
    print("[1] 가상 효용 u(행동,목표): " + ", ".join("%s=%.1f" % (k, v) for k, v in U.items()))
    a = U["nAnG"] - U["AnG"]
    b = U["AG"] - U["nAG"]
    print("[2] p* = %.2f / (%.2f + %.2f) = %.3f" % (a, a, b, p_star()))
    ps = p_star()
    print("[3] p=p*에서 실행의 기대 효용 %.3f, 가만히의 기대 효용 %.3f" % (eu_act(ps), eu_none(ps)))
    print("[4] 묻기를 넣으면 경계가 둘: 가만히-묻기 %.3f, 묻기-실행 %.3f" % (p_none_dialog(), p_dialog_act()))
    print("[5] p별 선택")
    for p in (0.50, 0.65, 0.75, 0.80, 0.95):
        print("  p=%.2f -> %s" % (p, decide(p)))
    print("[6] 효용을 바꾸면 p*가 움직인다(가상)")
    for label, key, val in (
        ("기준", None, None),
        ("틀린 환불 실행 u(A,nG) 0.0 -> 0.5", "AnG", 0.5),
        ("옳은 환불을 미룬 u(nA,G) 0.6 -> 0.3", "nAG", 0.3),
        ("옳은 환불을 미룬 u(nA,G) 0.6 -> 0.9", "nAG", 0.9),
    ):
        u = dict(U)
        if key:
            u[key] = val
        print("  %s: p*=%.3f" % (label, p_star(u)))


if __name__ == "__main__":
    main()
