"""15장 실습: 식별 전략 선택 지도. 같은 데이터에 방법을 바꿔 대 보고, 데이터 구조에서 후보를 고른다."""
import random

TRUE_EFFECT = 1500   # 가상 쿠폰 효과(원)
CUTOFF = 60          # 점수 60점 이상에게만 쿠폰(가상)

CHAPTERS = {
    2: "무작위 실험 설계와 비순응", 3: "DAG와 통제변수", 4: "성향점수·매칭·가중", 5: "이중강건 추정",
    6: "이중차분", 7: "시차 도입과 이벤트스터디", 8: "회귀불연속", 9: "도구변수와 LATE",
    10: "합성통제", 11: "간섭과 SUTVA", 12: "스위치백과 세척기간", 13: "CATE와 업리프트",
    14: "민감도·위약검정·외적타당도",
}


# ---------- 같은 데이터, 여러 방법 ----------
def make_cutoff_data(n=20000, seed=3):
    """점수가 높을수록 평소 매출도 크고 매출이 더 빨리 자란다. 쿠폰은 점수 60 이상에게만."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        s = rng.uniform(0, 100)
        t = int(s >= CUTOFF)
        pre = 10000 + 120 * s + rng.gauss(0, 2000)
        post = pre + 500 + 8 * s + TRUE_EFFECT * t + rng.gauss(0, 2000)
        rows.append({"s": s, "t": t, "pre": pre, "post": post})
    return rows


def _mean(xs):
    return sum(xs) / len(xs)


def before_after_treated(rows):
    tr = [r for r in rows if r["t"]]
    return _mean([r["post"] - r["pre"] for r in tr])


def naive_post_diff(rows):
    return (_mean([r["post"] for r in rows if r["t"]]) - _mean([r["post"] for r in rows if not r["t"]]))


def did_estimate(rows):
    tr = [r["post"] - r["pre"] for r in rows if r["t"]]
    co = [r["post"] - r["pre"] for r in rows if not r["t"]]
    return _mean(tr) - _mean(co)


def solve(a, b):
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(m[r][c]))
        m[c], m[p] = m[p], m[c]
        for r in range(n):
            if r != c:
                f = m[r][c] / m[c][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [m[i][n] / m[i][i] for i in range(n)]


def rdd_local_linear(rows, bandwidth):
    """기준선 양쪽의 직선을 따로 맞추고, 기준선에서의 점프를 읽는다. 열: 1, 처치, (s-60), 처치*(s-60)."""
    sub = [r for r in rows if abs(r["s"] - CUTOFF) <= bandwidth]
    k = 4
    xtx = [[0.0] * k for _ in range(k)]
    xty = [0.0] * k
    for r in sub:
        x = [1.0, r["t"], r["s"] - CUTOFF, r["t"] * (r["s"] - CUTOFF)]
        for i in range(k):
            xty[i] += x[i] * r["post"]
            for j in range(k):
                xtx[i][j] += x[i] * x[j]
    beta = solve(xtx, xty)
    sse = 0.0
    for r in sub:
        x = [1.0, r["t"], r["s"] - CUTOFF, r["t"] * (r["s"] - CUTOFF)]
        sse += (r["post"] - sum(b * v for b, v in zip(beta, x))) ** 2
    sigma2 = sse / (len(sub) - k)
    inv_11 = solve(xtx, [0.0, 1.0, 0.0, 0.0])[1]   # (X'X)^-1 의 (1,1) 성분
    return beta[1], len(sub), (sigma2 * inv_11) ** 0.5


# ---------- 데이터 구조 -> 후보 ----------
def recommend(facts):
    """facts의 키: assignment(random|cutoff|staggered|one_unit|self_selected), pre_period(bool),
    untreated_group(bool), full_compliance(bool), spillover(bool), instrument(bool),
    covariates_cover_selection(bool), effect_varies_question(bool).
    반환: (먼저 막는 문제 목록, 후보 [(장, 이유)])."""
    blockers, picks = [], []
    if facts.get("spillover"):
        blockers.append((11, "처치가 이웃에게 번지면 비교 집단이 오염돼요. 단위를 묶거나 시간으로 번갈아 켤 수 있는지부터 봐요."))
        if facts.get("can_toggle"):
            picks.append((12, "시간을 번갈아 켤 수 있어요. 이월효과가 사라질 세척기간을 정해요."))
    a = facts.get("assignment")
    if a == "random":
        picks.append((2, "무작위로 배정했으니 배정 기준으로 먼저 비교해요(ITT)."))
        if not facts.get("full_compliance", True):
            picks.append((9, "배정과 실제 사용이 다르면 배정을 도구로 쓰는 LATE를 후보에 둬요."))
    elif a == "cutoff":
        picks.append((8, "기준선 규칙으로 배정했으니 기준선 바로 위아래를 비교해요."))
    elif a == "staggered":
        picks.append((7, "도입일이 제각각이면 이미 처치된 집단을 비교군으로 쓰지 않는 방법이 필요해요."))
    elif a == "one_unit":
        if facts.get("pre_period") and facts.get("untreated_group"):
            picks.append((10, "처치 단위가 하나면 비교 단위를 가중해서 만들어요."))
        else:
            blockers.append((10, "처치 단위가 하나인데 처치 전 기간이나 비교 단위 풀이 없으면 합성할 재료가 없어요."))
    elif a == "self_selected":
        if facts.get("instrument"):
            picks.append((9, "처치 선택을 밀어 주지만 결과에는 직접 닿지 않는 도구가 있으면 후보예요."))
        if facts.get("pre_period") and facts.get("untreated_group"):
            picks.append((6, "처치 전후와 비교 집단이 있으면 이중차분 후보예요. 평행추세를 그려 확인해요."))
        if facts.get("covariates_cover_selection"):
            picks.append((3, "선택에 쓰인 변수를 모두 관측했다면 DAG로 통제할 변수를 고르고,"))
            picks.append((4, "성향점수나 매칭으로 비교하고"))
            picks.append((5, "결과모형과 함께 쓰는 이중강건 추정으로 안전장치를 둬요."))
        if not picks:
            blockers.append((3, "선택 이유를 관측하지 못했고 기준선·도구·전후 비교도 없어요. 지금 데이터로는 효과를 말하기 어려워요."))
    if facts.get("effect_varies_question") and picks:
        picks.append((13, "누구에게 효과가 큰지는 사전에 정한 하위집단부터 보고, 탐색한 것은 보류 데이터로 확인해요."))
    if picks:
        picks.append((14, "고른 방법의 가정이 깨졌을 때를 민감도·위약검정으로 확인해요."))
    return blockers, picks


def main():
    rows = make_cutoff_data()
    print(f"진짜 효과 {TRUE_EFFECT}원")
    print(f"처치군 전후 비교 {before_after_treated(rows):.0f}원")
    print(f"사후 단순 차이 {naive_post_diff(rows):.0f}원")
    print(f"이중차분(60점 미만을 비교군으로) {did_estimate(rows):.0f}원")
    for bw in (5, 10, 20):
        est, n, se = rdd_local_linear(rows, bw)
        print(f"회귀불연속 대역폭 {bw}: {est:.0f}원 (표준오차 {se:.0f}, 표본 {n}명)")
    cases = {
        "무작위+사용 불일치": dict(assignment="random", full_compliance=False),
        "점수 기준 쿠폰": dict(assignment="cutoff"),
        "지역별 도입일 다름": dict(assignment="staggered"),
        "한 지역만 처치, 비교 풀 없음": dict(assignment="one_unit", pre_period=True, untreated_group=False),
        "자기선택, 아무 장치 없음": dict(assignment="self_selected"),
    }
    for name, f in cases.items():
        b, p = recommend(f)
        print(name, "->", [c for c, _ in p], "막힘", [c for c, _ in b])


if __name__ == "__main__":
    main()
