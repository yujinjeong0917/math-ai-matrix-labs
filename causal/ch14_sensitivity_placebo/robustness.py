"""14장 실습: 민감도 분석, 위약검정, 외적타당도. 표준 라이브러리만 쓴다."""
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
        p = 1 / (1 + math.exp(-(-0.3 + 0.8 * h + 0.9 * u)))
        t = int(rng.random() < p)
        y = 20000 + gamma_h * h + gamma_u * u + TRUE_EFFECT * t + rng.gauss(0, 5000)
        rows.append({"h": h, "u": u, "t": t, "y": y})
    return rows


def _mean(xs):
    return sum(xs) / len(xs)


def naive_diff(rows):
    return _mean([r["y"] for r in rows if r["t"]]) - _mean([r["y"] for r in rows if not r["t"]])


def stratified_diff(rows, key="h", field="y"):
    """관측 변수 h로 층을 나눠 층별 차이를 층 크기로 가중 평균한다."""
    total, n = 0.0, len(rows)
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
    return {(g, d): estimate - omitted_variable_bias(g, d) for g in gammas for d in deltas}


def breakeven_delta(estimate, gamma):
    """추정값을 0으로 만들려면 처치군-비교군 U 평균 차이가 얼마나 커야 하나."""
    return estimate / gamma


# ---------- B. 위약검정 ----------
def make_panel(effect, extra_slope, regions=20, periods=6, start=3, seed=5, noise=1.0):
    """지역 r, 기간 t의 평균 주문지수. 지역 0만 처치(start 기간부터)."""
    rng = random.Random(seed)
    fe = [rng.gauss(100, 15) for _ in range(regions)]
    y = [[0.0] * periods for _ in range(regions)]
    for r in range(regions):
        for t in range(periods):
            y[r][t] = (fe[r] + 2.0 * t + (extra_slope * t if r == 0 else 0)
                       + (effect if (r == 0 and t >= start) else 0) + rng.gauss(0, noise))
    return y


def did_vs_donors(y, unit, donors, pre, post):
    def avg(r, ts):
        return _mean([y[r][t] for t in ts])
    own = avg(unit, post) - avg(unit, pre)
    ctrl = _mean([avg(r, post) - avg(r, pre) for r in donors])
    return own - ctrl


def placebo_report(y, start=3):
    periods = len(y[0])
    donors = list(range(1, len(y)))
    pre, post = list(range(0, start)), list(range(start, periods))
    actual = did_vs_donors(y, 0, donors, pre, post)
    # 공간 위약: 처치받지 않은 지역을 하나씩 처치 지역인 척 한다.
    space = [did_vs_donors(y, r, [d for d in donors if d != r], pre, post) for r in donors]
    space_rank = 1 + sum(abs(s) >= abs(actual) for s in space)  # 실제 값이 몇 번째로 큰가
    # 시간 위약: 처치 전 기간만 써서 가짜 처치 시점(마지막 처치 전 기간)을 만든다.
    fpre, fpost = list(range(0, start - 1)), [start - 1]
    time_est = did_vs_donors(y, 0, donors, fpre, fpost)
    time_donor = [did_vs_donors(y, r, [d for d in donors if d != r], fpre, fpost) for r in donors]
    time_rank = 1 + sum(abs(s) >= abs(time_est) for s in time_donor)
    return {"did": actual, "space_rank": space_rank, "n_units": len(y),
            "time_placebo": time_est, "time_rank": time_rank}


# ---------- C. 외적타당도(다른 지역으로 옮기기) ----------
TAU_NEW, TAU_OLD = 3000, 600  # 가상: 신규/기존 고객 효과(원)


def make_experiment(n, new_share, seed, multiplier=1.0):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        new = int(rng.random() < new_share)
        t = int(rng.random() < 0.5)
        tau = multiplier * (TAU_NEW if new else TAU_OLD)
        rows.append({"new": new, "t": t, "y": 20000 + tau * t + rng.gauss(0, 6000)})
    return rows


def effect(rows):
    a = [r["y"] for r in rows if r["t"]]
    b = [r["y"] for r in rows if not r["t"]]
    return _mean(a) - _mean(b)


def transport(source_rows, target_new_share):
    """원천 지역의 신규/기존별 효과에 목표 지역의 신규 비중을 곱해 다시 평균낸다."""
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
    carry = ate_a
    moved = transport(a, 0.2)
    truth_b = 0.2 * TAU_NEW + 0.8 * TAU_OLD
    print(f"A 실험 ATE {ate_a:.0f}원 (A 진짜 {0.6 * TAU_NEW + 0.4 * TAU_OLD:.0f}원)")
    print(f"B(신규 20%)에 그대로 옮김 {carry:.0f}원 / 신규 비중으로 다시 가중 {moved:.0f}원 / B 진짜 {truth_b:.0f}원")
    print(f"B에 숨은 요인이 효과를 절반으로 줄이면 진짜 {0.5 * truth_b:.0f}원, 다시 가중한 값은 {moved:.0f}원 그대로")


if __name__ == "__main__":
    main()
