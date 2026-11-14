"""10장 합성통제. 표준 라이브러리만 쓴다. 가상 온라인 마켓: 지역 A에만 무료배송을 켠 뒤 주간 주문지수.

처치 지역 1개, 대조 후보(도너) 지역 8개, 처치 전 20주, 후 8주. 진짜 효과는 후 기간 매주 +10(가상 지수).
가중치는 '합이 1이고 음수가 아닌' 조건에서 처치 전 20주를 가장 잘 맞추는 조합을 격자·좌표 탐색으로 찾는다.
"""
import random
from statistics import mean

T0, T1, J = 20, 8, 8
TRUE_EFFECT = 10.0


def factors():
    f1 = [t / 6.0 for t in range(T0 + T1)]  # 완만한 성장 추세
    f2 = [3.0 * ((t % 7) - 3) / 3.0 for t in range(T0 + T1)]  # 7주 주기 파동
    return f1, f2


def make_panel(seed=3, in_hull=True):
    """반환: (처치 지역 경로, [도너 경로 8개]). 마지막 T1주에 처치 지역만 +10."""
    rng = random.Random(seed)
    f1, f2 = factors()
    donors = []
    loads = []
    for j in range(J):
        level = 80 + 5 * j  # 80..115
        a = 4 + 1.0 * j + rng.uniform(-0.5, 0.5)  # 추세 기울기 4..11
        b = 2 + 0.5 * j + rng.uniform(-0.5, 0.5)  # 파동 크기
        loads.append((level, a, b))
        donors.append([level + a * f1[t] + b * f2[t] + rng.gauss(0, 0.6) for t in range(T0 + T1)])
    if in_hull:
        mix = {1: 0.5, 3: 0.3, 6: 0.2}  # 도너 1,3,6번의 볼록결합
        lv = sum(w * loads[j][0] for j, w in mix.items())
        a = sum(w * loads[j][1] for j, w in mix.items())
        b = sum(w * loads[j][2] for j, w in mix.items())
    else:
        # 모든 도너보다 수준도 높고 추세도 가파른 지역(볼록껍질 밖)
        lv, a, b = 150.0, 20.0, 2.0
    treated = [lv + a * f1[t] + b * f2[t] + rng.gauss(0, 0.6) for t in range(T0 + T1)]
    for t in range(T0, T0 + T1):
        treated[t] += TRUE_EFFECT
    return treated, donors


def sse(w, y, donors, t_range):
    s = 0.0
    for t in t_range:
        s += (y[t] - sum(w[j] * donors[j][t] for j in range(len(w)))) ** 2
    return s


def fit_weights(y, donors, t_range=None):
    """합=1, 음수 없음 조건의 최소제곱을 '한 도너에서 다른 도너로 가중치 옮기기' 좌표 탐색으로 푼다."""
    t_range = list(range(T0)) if t_range is None else t_range
    n = len(donors)
    w = [1.0 / n] * n
    best = sse(w, y, donors, t_range)
    step = 0.1
    while step > 1e-5:
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j or w[i] < step:
                    continue
                w[i] -= step
                w[j] += step
                s = sse(w, y, donors, t_range)
                if s < best - 1e-12:
                    best, improved = s, True
                else:
                    w[i] += step
                    w[j] -= step
        if not improved:
            step /= 2
    return w


def rmspe(y, synth, rng_):
    return (mean((y[t] - synth[t]) ** 2 for t in rng_)) ** 0.5


def synth_path(w, donors):
    return [sum(w[j] * donors[j][t] for j in range(len(w))) for t in range(T0 + T1)]


def analyze(treated, donors):
    w = fit_weights(treated, donors)
    s = synth_path(w, donors)
    pre = rmspe(treated, s, range(T0))
    post = rmspe(treated, s, range(T0, T0 + T1))
    gap = mean(treated[t] - s[t] for t in range(T0, T0 + T1))
    return {"weights": w, "pre_rmspe": pre, "post_rmspe": post, "effect": gap}


def equal_weight_effect(treated, donors):
    """대조 후보 8개의 단순 평균을 쓴 이중차분. 처치 전 평균 차이를 상수로 빼 준다."""
    avg = [mean(d[t] for d in donors) for t in range(T0 + T1)]
    pre_gap = mean(treated[t] - avg[t] for t in range(T0))
    post_gap = mean(treated[t] - avg[t] for t in range(T0, T0 + T1))
    return post_gap - pre_gap


def placebo_ratios(treated, donors):
    """가짜 처치: 도너를 하나씩 처치 지역인 척하고(진짜 처치 지역은 풀에서 뺀다) 같은 방법을 돌린다.
    반환: 처치 지역과 도너들의 (후 RMSPE / 전 RMSPE)."""
    units = [treated] + donors
    ratios = []
    for k, y in enumerate(units):
        pool = [u for i, u in enumerate(units) if i != k and not (k != 0 and i == 0)]
        w = fit_weights(y, pool)
        s = synth_path(w, pool)
        pre = rmspe(y, s, range(T0))
        post = rmspe(y, s, range(T0, T0 + T1))
        ratios.append(post / max(pre, 1e-9))
    return ratios


def report():
    out = {}
    for name, hull in (("in_hull", True), ("out_hull", False)):
        tr, dn = make_panel(in_hull=hull)
        a = analyze(tr, dn)
        a["equal_weight_did"] = equal_weight_effect(tr, dn)
        a["max_donor_pre_mean"] = max(mean(d[t] for t in range(T0)) for d in dn)
        a["treated_pre_mean"] = mean(tr[t] for t in range(T0))
        r = placebo_ratios(tr, dn)
        a["placebo_rank"] = 1 + sum(1 for x in r[1:] if x >= r[0])
        a["placebo_ratio_treated"] = r[0]
        a["placebo_ratio_max_donor"] = max(r[1:])
        sp = synth_path(a["weights"], dn)
        a["pre_mean_gap"] = mean(tr[t] - sp[t] for t in range(T0))
        a["placebo_ratios_donors"] = [round(x, 2) for x in r[1:]]
        a["placebo_n_donors_above_treated"] = sum(1 for x in r[1:] if x >= r[0])
        if hull:
            design = [0.5 * dn[1][t] + 0.3 * dn[3][t] + 0.2 * dn[6][t] for t in range(T0 + T1)]  # 설계 가중치(2번 0.5, 4번 0.3, 7번 0.2)
            a["design_weights_pre_rmspe"] = rmspe(tr, design, range(T0))
            a["design_vs_fitted_path_rmse"] = rmspe(design, sp, range(T0 + T1))
        a["effect_minus_pre_gap_minus_true"] = a["effect"] - a["pre_mean_gap"] - TRUE_EFFECT  # 후 간격 중 처치 전 수준 차이와 진짜 효과를 뺀 추세 몫
        a["equal_weight_did_minus_true"] = a["equal_weight_did"] - TRUE_EFFECT
        out[name] = a
    return out


if __name__ == "__main__":
    for name, a in report().items():
        print(name)
        for k, v in a.items():
            if k == "weights":
                print("  weights", [round(x, 3) for x in v])
            else:
                print("  ", k, round(v, 3) if isinstance(v, float) else v)
