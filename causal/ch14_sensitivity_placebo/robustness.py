"""14장 실습: 민감도 분석, 위약검정, 외적타당도. 표준 라이브러리만 쓴다.

무엇을 재는가: (A) 관측하지 못한 교란 U가 있을 때 추정값이 얼마나 틀어지는지(편향 gamma x delta)와 그것을 0으로 만들 U 차이(breakeven),
(B) 위약검정(placebo test, 효과가 없어야 하는 곳에 같은 방법을 적용): 시간 위약은 처치 직전 기간을 가짜 처치 시점으로, 공간 위약은 처치 안 된 지역을 처치 지역인 척 한다,
(C) 외적타당도: 다른 지역으로 효과를 옮길 때 관측된 효과 수정변수(신규 비중)로 다시 가중하면 얼마나 맞는지.
입력: make_observational, make_panel, make_experiment이 만드는 가상 데이터. 금액 단위는 가상의 원, 주문지수는 가상 단위. 진짜 효과는 TRUE_EFFECT = 1500원.
출력: 단순 차이, 층화 추정, 민감도 표, 위약 순위, 옮긴 추정. 테스트의 기대 조건이다(장 본문 숫자와의 대응은 README로 확인 필요).
"""
import math
import random

TRUE_EFFECT = 1500  # 가상 세계의 진짜 쿠폰 효과(원)


# ---------- A. 관측 데이터와 숨은 교란(민감도) ----------
def make_observational(n=20000, seed=11, gamma_u=4000, gamma_h=6000):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        h = int(rng.random() < 0.3)          # 관측되는 변수: 평소 많이 사는 고객
        u = rng.gauss(0, 1)                   # 관측되지 않는 변수: 충성도(표준편차 단위)
        # 쿠폰을 받을 확률이 h와 u에 모두 오른다(로지스틱). u가 처치와 결과 둘 다에 영향을 주는 숨은 교란이다
        p = 1 / (1 + math.exp(-(-0.3 + 0.8 * h + 0.9 * u)))
        t = int(rng.random() < p)
        y = 20000 + gamma_h * h + gamma_u * u + TRUE_EFFECT * t + rng.gauss(0, 5000)
        rows.append({"h": h, "u": u, "t": t, "y": y})
    return rows


def _mean(xs):
    """산술평균(내부용)."""
    return sum(xs) / len(xs)


def naive_diff(rows):
    """처치군 평균 - 비교군 평균. h와 u 둘 다의 교란이 섞여 진짜 1500보다 크게 나온다."""
    return _mean([r["y"] for r in rows if r["t"]]) - _mean([r["y"] for r in rows if not r["t"]])


def stratified_diff(rows, key="h", field="y"):
    """관측 변수 h로 층을 나눠 층별 차이를 층 크기로 가중 평균한다."""
    total, n = 0.0, len(rows)
    # key는 0/1 값을 갖는 관측 변수만 가능하다(층 두 개). 층별 차이를 층 인원 비율로 가중해 합친다
    for v in (0, 1):
        sub = [r for r in rows if r[key] == v]
        a = [r[field] for r in sub if r["t"]]
        b = [r[field] for r in sub if not r["t"]]
        total += len(sub) / n * (_mean(a) - _mean(b))
    return total


def omitted_variable_bias(gamma, delta):
    """숨은 변수 U 한 단위가 결과를 gamma만큼 바꾸고, 처치군의 U 평균이 delta만큼 높으면 편향은 gamma*delta."""
    return gamma * delta


def sensitivity_table(estimate, gammas, deltas):
    """gamma(숨은 변수의 결과 영향)와 delta(처치군-비교군 숨은 변수 차이) 조합마다 편향을 뺀 추정. 키는 (gamma, delta)."""
    return {(g, d): estimate - omitted_variable_bias(g, d) for g in gammas for d in deltas}


def breakeven_delta(estimate, gamma):
    """추정값을 0으로 만들려면 처치군-비교군 U 평균 차이가 얼마나 커야 하나."""
    return estimate / gamma


# ---------- B. 위약검정 ----------
def make_panel(effect, extra_slope, regions=20, periods=6, start=3, seed=5, noise=1.0):
    """지역 r, 기간 t의 평균 주문지수. 지역 0만 처치(start 기간부터).
    extra_slope가 0이 아니면 처치 지역만 더 가파르게 자라는 "평행추세 위반"이 된다(효과가 0이어도 DiD가 양수로 나온다)."""
    rng = random.Random(seed)
    fe = [rng.gauss(100, 15) for _ in range(regions)]  # 지역 고정효과(수준 차이)
    y = [[0.0] * periods for _ in range(regions)]
    for r in range(regions):
        for t in range(periods):
            # 모든 지역 공통 추세 2.0 x t. 처치 지역만 extra_slope x t가 더해질 수 있다
            y[r][t] = (fe[r] + 2.0 * t + (extra_slope * t if r == 0 else 0)
                       + (effect if (r == 0 and t >= start) else 0) + rng.gauss(0, noise))
    return y


def did_vs_donors(y, unit, donors, pre, post):
    """단위 unit의 전후 변화 - 도너 지역들의 평균 전후 변화(이중차분). pre, post는 기간 인덱스 목록."""
    def avg(r, ts):
        return _mean([y[r][t] for t in ts])
    own = avg(unit, post) - avg(unit, pre)
    ctrl = _mean([avg(r, post) - avg(r, pre) for r in donors])
    return own - ctrl


def placebo_report(y, start=3):
    """실제 DiD, 공간 위약 순위, 시간 위약 추정과 순위. 순위 1이면 "가짜 처치들보다 실제가 가장 크다"는 뜻이다."""
    periods = len(y[0])
    donors = list(range(1, len(y)))
    pre, post = list(range(0, start)), list(range(start, periods))
    actual = did_vs_donors(y, 0, donors, pre, post)
    # 공간 위약: 처치받지 않은 지역을 하나씩 처치 지역인 척 한다.
    space = [did_vs_donors(y, r, [d for d in donors if d != r], pre, post) for r in donors]
    space_rank = 1 + sum(abs(s) >= abs(actual) for s in space)  # 실제 값이 몇 번째로 큰가(절댓값 기준)
    # 시간 위약: 처치 전 기간만 써서 가짜 처치 시점(마지막 처치 전 기간)을 만든다.
    # 가짜 처치 시점을 진짜 시점 직전 기간(start - 1)으로 당긴다. 이 구간에는 진짜 효과가 없어야 하므로 추정이 0 근처여야 정상이다
    fpre, fpost = list(range(0, start - 1)), [start - 1]
    time_est = did_vs_donors(y, 0, donors, fpre, fpost)
    time_donor = [did_vs_donors(y, r, [d for d in donors if d != r], fpre, fpost) for r in donors]
    time_rank = 1 + sum(abs(s) >= abs(time_est) for s in time_donor)
    return {"did": actual, "space_rank": space_rank, "n_units": len(y),
            "time_placebo": time_est, "time_rank": time_rank}


# ---------- C. 외적타당도(다른 지역으로 옮기기) ----------
TAU_NEW, TAU_OLD = 3000, 600  # 가상: 신규/기존 고객 효과(원). 신규 비중이 다른 지역으로 옮기면 평균 효과가 달라진다


def make_experiment(n, new_share, seed, multiplier=1.0):
    """지역 하나의 무작위 실험. new_share는 그 지역의 신규 고객 비중, multiplier는 효과 전체의 배율(숨은 요인 시나리오용)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        new = int(rng.random() < new_share)
        t = int(rng.random() < 0.5)
        tau = multiplier * (TAU_NEW if new else TAU_OLD)
        rows.append({"new": new, "t": t, "y": 20000 + tau * t + rng.gauss(0, 6000)})
    return rows


def effect(rows):
    """처치군 평균 - 비교군 평균(무작위 배정이라 효과 추정)."""
    a = [r["y"] for r in rows if r["t"]]
    b = [r["y"] for r in rows if not r["t"]]
    return _mean(a) - _mean(b)


def transport(source_rows, target_new_share):
    """원천 지역의 신규/기존별 효과에 목표 지역의 신규 비중을 곱해 다시 평균낸다.
    가정: 효과를 바꾸는 요인이 신규 여부뿐이고 그 외 숨은 요인은 없다. 이 가정이 깨지면 맞지 않는다."""
    e_new = effect([r for r in source_rows if r["new"]])
    e_old = effect([r for r in source_rows if not r["new"]])
    return target_new_share * e_new + (1 - target_new_share) * e_old


def main():
    rows = make_observational()
    naive, strat = naive_diff(rows), stratified_diff(rows)
    # 지금은 U를 알고 있으니 편향 공식이 맞는지 확인할 수 있다(실제 분석에서는 못 본다).
    delta = 0.0
    for v in (0, 1):
        sub = [r for r in rows if r["h"] == v]
        du = _mean([r["u"] for r in sub if r["t"]]) - _mean([r["u"] for r in sub if not r["t"]])
        delta += len(sub) / len(rows) * du
    print(f"진짜 효과 {TRUE_EFFECT}원 / 단순 차이 {naive:.0f}원 / h로 층화 {strat:.0f}원")
    print(f"숨은 U의 평균 차이 delta={delta:.3f}, 편향 공식 4000*delta={omitted_variable_bias(4000, delta):.0f}원, "
          f"숨은 요인 보정 추정 {strat - omitted_variable_bias(4000, delta):.0f}원")
    print("민감도 표(층화 추정 - gamma*delta):")
    tab = sensitivity_table(strat, (1000, 2000, 4000), (0.2, 0.5, 0.8))
    for (g, d), v in tab.items():
        print(f"  gamma={g}, delta={d}: {v:.0f}원")
    for g in (2000, 4000):
        print(f"  gamma={g}에서 추정을 0으로 만드는 delta = {breakeven_delta(strat, g):.2f}")
    print(f"비교 기준: 관측 변수 h를 통제하자 추정이 {naive:.0f}원에서 {strat:.0f}원으로 {naive - strat:.0f}원 줄었다")
    for name, eff, slope in (("평행추세 세계", 10.0, 0.0), ("처치 지역만 가파른 세계", 0.0, 3.0)):
        rep = placebo_report(make_panel(eff, slope))
        print(f"{name}: DiD {rep['did']:.2f}, 공간 위약 순위 {rep['space_rank']}/{rep['n_units']}, "
              f"시간 위약 {rep['time_placebo']:.2f}, 시간 위약 순위 {rep['time_rank']}/{rep['n_units']}")
    a = make_experiment(20000, 0.6, seed=21)
    ate_a = effect(a)
    carry = ate_a  # 신규 비중이 다른 B에 아무 조정 없이 옮기는 경우(비교용)
    moved = transport(a, 0.2)
    truth_b = 0.2 * TAU_NEW + 0.8 * TAU_OLD
    print(f"A 실험 ATE {ate_a:.0f}원 (A 진짜 {0.6 * TAU_NEW + 0.4 * TAU_OLD:.0f}원)")
    print(f"B(신규 20%)에 그대로 옮김 {carry:.0f}원 / 신규 비중으로 다시 가중 {moved:.0f}원 / B 진짜 {truth_b:.0f}원")
    print(f"B에 숨은 요인이 효과를 절반으로 줄이면 진짜 {0.5 * truth_b:.0f}원, 다시 가중한 값은 {moved:.0f}원 그대로")


if __name__ == "__main__":
    main()
