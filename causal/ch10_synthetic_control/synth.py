"""10장 합성통제. 표준 라이브러리만 쓴다. 가상 온라인 마켓: 지역 A에만 무료배송을 켠 뒤 주간 주문지수.

처치 지역 1개, 대조 후보(도너) 지역 8개, 처치 전 20주, 후 8주. 진짜 효과는 후 기간 매주 +10(가상 지수).
가중치는 '합이 1이고 음수가 아닌' 조건에서 처치 전 20주를 가장 잘 맞추는 조합을 격자·좌표 탐색으로 찾는다.

무엇을 재는가: 처치 지역의 처치 후 경로에서 "도너 가중 평균으로 만든 가상의 처치 지역(합성 대조)"을 뺀 값을 효과로 읽는다.
처치 전 맞춤(pre_rmspe, RMSPE = 제곱평균제곱근 예측오차)이 좋을 때만 믿을 수 있고, 처치 지역이 도너의 볼록껍질 밖이면 맞추지 못해 틀린다.
가짜 처치(placebo) 검정은 도너를 하나씩 처치 지역인 척하고 후/전 RMSPE 비율의 순위를 본다.
입력: make_panel(seed, in_hull)의 (처치 경로, 도너 경로 8개). 길이는 처치 전 T0 + 후 T1 = 28주. 출력: analyze/report의 dict.
테스트의 기대값(처치 전 RMSPE < 1, 효과 약 10, 가짜 처치 순위 1 등)이 장 본문의 표와 대응한다. 숫자는 모두 가상의 주간 주문지수.
"""
import random
from statistics import mean

T0, T1, J = 20, 8, 8
TRUE_EFFECT = 10.0


def factors():
    """모든 지역이 공유하는 두 공통 요인(시간에 따른 모양). 지역마다 적재량(loads)이 달라 경로가 다르다."""
    f1 = [t / 6.0 for t in range(T0 + T1)]  # 완만한 성장 추세
    # 계수 3.0과 분모 3.0이 약분돼 사실상 (t % 7) - 3. 값의 범위 -3..3인 톱니 모양 주기
    f2 = [3.0 * ((t % 7) - 3) / 3.0 for t in range(T0 + T1)]  # 7주 주기 파동
    return f1, f2


def make_panel(seed=3, in_hull=True):
    """반환: (처치 지역 경로, [도너 경로 8개]). 마지막 T1주에 처치 지역만 +10."""
    # 요인 모형: 지역 j의 주간값 = 수준 + 추세 기울기 x f1 + 파동 크기 x f2 + 잡음
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
        # 도너 인덱스 1, 3, 6(0부터 센 번호; 본문의 2번, 4번, 7번)의 볼록결합으로 처치 지역을 만든다(정답 가중치가 존재)
        mix = {1: 0.5, 3: 0.3, 6: 0.2}  # 도너 1,3,6번의 볼록결합
        lv = sum(w * loads[j][0] for j, w in mix.items())
        a = sum(w * loads[j][1] for j, w in mix.items())
        b = sum(w * loads[j][2] for j, w in mix.items())
    else:
        # 모든 도너보다 수준도 높고 추세도 가파른 지역(볼록껍질 밖)
        lv, a, b = 150.0, 20.0, 2.0  # 수준 150은 도너 최댓값(115)을, 추세 20은 도너 기울기 범위(약 3.5~11.5)를 넘는다(파동 크기 2.0은 범위 안). 그래서 가중 평균으로 흉내 낼 수 없다
    treated = [lv + a * f1[t] + b * f2[t] + rng.gauss(0, 0.6) for t in range(T0 + T1)]
    for t in range(T0, T0 + T1):  # 처치 후 8주(인덱스 20..27)에만 효과가 얹힌다
        treated[t] += TRUE_EFFECT
    return treated, donors


def sse(w, y, donors, t_range):
    """가중 평균 도너 경로와 y의 잔차 제곱합(구간 t_range만)."""
    s = 0.0
    for t in t_range:
        s += (y[t] - sum(w[j] * donors[j][t] for j in range(len(w)))) ** 2
    return s


def fit_weights(y, donors, t_range=None):
    """합=1, 음수 없음 조건의 최소제곱을 '한 도너에서 다른 도너로 가중치 옮기기' 좌표 탐색으로 푼다."""
    # 기본은 처치 전 구간만 맞춘다(처치 후 값을 맞추면 효과가 지워진다)
    t_range = list(range(T0)) if t_range is None else t_range
    n = len(donors)
    w = [1.0 / n] * n  # 균등 가중치에서 출발
    best = sse(w, y, donors, t_range)
    # 가중치를 i에서 빼 j에 더하는 이동은 합 1을 보존한다. w[i] < step이면 음수가 되므로 건너뛴다(음수 금지 제약)
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
                if s < best - 1e-12:  # 1e-12는 부동소수점 잡음으로 인한 무한 반복 방지
                    best, improved = s, True
                else:
                    w[i] += step
                    w[j] -= step
        if not improved:  # 이 보폭으로 더 줄일 수 없으면 보폭을 절반으로 줄여 세밀하게 찾는다(1e-5까지)
            step /= 2
    return w


def rmspe(y, synth, rng_):
    """구간 rng_에서 실제와 합성 경로의 제곱평균제곱근 오차. 같은 척도(주문지수)의 오차 크기."""
    return (mean((y[t] - synth[t]) ** 2 for t in rng_)) ** 0.5


def synth_path(w, donors):
    """가중치 w로 만든 합성 대조 경로(전체 28주)."""
    return [sum(w[j] * donors[j][t] for j in range(len(w))) for t in range(T0 + T1)]


def analyze(treated, donors):
    """가중치 적합 -> 합성 경로 -> 처치 전/후 RMSPE와 효과(후 8주 평균 간격). 효과는 "맞춤이 좋을 때만" 의미가 있다."""
    w = fit_weights(treated, donors)
    s = synth_path(w, donors)
    pre = rmspe(treated, s, range(T0))
    post = rmspe(treated, s, range(T0, T0 + T1))
    gap = mean(treated[t] - s[t] for t in range(T0, T0 + T1))  # 후 8주 평균 간격이 효과 추정
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
    units = [treated] + donors  # 인덱스 0이 진짜 처치 지역
    ratios = []
    for k, y in enumerate(units):
        # 대조 풀에서 자기 자신을 빼고, 도너를 가짜 처치로 볼 때는 진짜 처치 지역(i == 0)도 뺀다
        pool = [u for i, u in enumerate(units) if i != k and not (k != 0 and i == 0)]
        w = fit_weights(y, pool)
        s = synth_path(w, pool)
        pre = rmspe(y, s, range(T0))
        post = rmspe(y, s, range(T0, T0 + T1))
        ratios.append(post / max(pre, 1e-9))  # 전 RMSPE가 0에 가까워 0으로 나누지 않도록 하한
    return ratios


def report():
    """볼록껍질 안(in_hull)과 밖(out_hull) 두 장면에 대해 추정·진단 값을 모아 dict로 돌려준다."""
    out = {}
    for name, hull in (("in_hull", True), ("out_hull", False)):
        tr, dn = make_panel(in_hull=hull)
        a = analyze(tr, dn)
        a["equal_weight_did"] = equal_weight_effect(tr, dn)
        a["max_donor_pre_mean"] = max(mean(d[t] for t in range(T0)) for d in dn)
        a["treated_pre_mean"] = mean(tr[t] for t in range(T0))
        r = placebo_ratios(tr, dn)
        # 가짜 처치 순위: 진짜 처치 지역의 후/전 비율이 도너 8개보다 크면 1위(순위 1)
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
