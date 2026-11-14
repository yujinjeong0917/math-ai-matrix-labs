"""2장 실습: 무작위 실험의 네 가지 함정. 모든 숫자는 가상(예시용)이다.

난수는 random.Random(seed).random()만 쓴다(셔플·샘플·가우스 함수는 쓰지 않는다).
"""
import math
import random


def normal(rng, sd=1.0):
    u1 = 1.0 - rng.random()
    u2 = rng.random()
    return sd * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def sd(xs):
    xs = list(xs)
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def random_half(rng, items):
    """items 중 정확히 절반을 무작위로 고른다(난수 키로 정렬)."""
    keyed = sorted(items, key=lambda _: rng.random())
    return set(keyed[: len(keyed) // 2])


# ---------- 1. 층화 무작위배정은 추정값의 흔들림을 줄인다 ----------
EFFECT = 2.0
N_HEAVY, N_LIGHT = 120, 280  # 단골 120명, 일반 280명(가상)
BASE = {"heavy": 30.0, "light": 10.0}
NOISE = 5.0


def one_experiment(rng, stratified):
    users = [("heavy", i) for i in range(N_HEAVY)] + [("light", i) for i in range(N_LIGHT)]
    if stratified:
        treated = set()
        for s in ("heavy", "light"):
            treated |= {(s, i) for (_, i) in random_half(rng, [u for u in users if u[0] == s])}
    else:
        treated = {u for u in users if rng.random() < 0.5}
    y_t = {"heavy": [], "light": []}
    y_c = {"heavy": [], "light": []}
    for u in users:
        t = u in treated
        y = BASE[u[0]] + (EFFECT if t else 0.0) + normal(rng, NOISE)
        (y_t if t else y_c)[u[0]].append(y)
    if stratified:  # 층별 차이를 층 크기로 가중
        n = N_HEAVY + N_LIGHT
        return (N_HEAVY / n) * (mean(y_t["heavy"]) - mean(y_c["heavy"])) + \
               (N_LIGHT / n) * (mean(y_t["light"]) - mean(y_c["light"]))
    all_t = y_t["heavy"] + y_t["light"]
    all_c = y_c["heavy"] + y_c["light"]
    return mean(all_t) - mean(all_c)


def stratification_demo(reps=2000, seed=21):
    rng = random.Random(seed)
    simple = [one_experiment(rng, False) for _ in range(reps)]
    strat = [one_experiment(rng, True) for _ in range(reps)]
    return {"simple_mean": mean(simple), "simple_sd": sd(simple),
            "strat_mean": mean(strat), "strat_sd": sd(strat), "reps": reps}


# ---------- 2. 군집 배정: 지역 단위로 나누면 독립 표본이 아니다 ----------
N_REGIONS, M_PER = 40, 50     # 지역 40곳, 지역당 고객 50명(가상)
REGION_SD, USER_SD = 3.0, 5.0  # 지역 효과 표준편차 3, 고객 잡음 5


def one_cluster_experiment(rng):
    treated_regions = random_half(rng, list(range(N_REGIONS)))
    yt, yc = [], []
    for r in range(N_REGIONS):
        shift = normal(rng, REGION_SD)
        for _ in range(M_PER):
            t = r in treated_regions
            y = 20.0 + shift + (EFFECT if t else 0.0) + normal(rng, USER_SD)
            (yt if t else yc).append(y)
    diff = mean(yt) - mean(yc)
    naive_se = math.sqrt(sd(yt) ** 2 / len(yt) + sd(yc) ** 2 / len(yc))
    return diff, naive_se


def cluster_demo(reps=500, seed=22):
    rng = random.Random(seed)
    res = [one_cluster_experiment(rng) for _ in range(reps)]
    diffs = [d for d, _ in res]
    naive_ses = [s for _, s in res]
    cover = mean(1.0 if abs(d - EFFECT) <= 1.96 * s else 0.0 for d, s in res)
    rho = REGION_SD ** 2 / (REGION_SD ** 2 + USER_SD ** 2)
    return {"mean_estimate": mean(diffs), "actual_sd": sd(diffs),
            "naive_se": mean(naive_ses), "naive_coverage": cover,
            "rho": rho, "design_effect": 1.0 + (M_PER - 1) * rho,
            "reps": reps}


# ---------- 3. SRM: 계획한 배정 비율과 실제 인원이 다를 때 ----------
def srm_pvalue(n_treat, n_control, ratio_treat=0.5):
    """자유도 1 카이제곱. p = 2(1 - Φ(√χ²)) = erfc(√(χ²/2))."""
    n = n_treat + n_control
    exp_t, exp_c = n * ratio_treat, n * (1 - ratio_treat)
    chi2 = (n_treat - exp_t) ** 2 / exp_t + (n_control - exp_c) ** 2 / exp_c
    return chi2, math.erfc(math.sqrt(chi2 / 2.0))


SRM_EFFECT = 1.0


def srm_demo(n=20000, seed=23, bug=True):
    """열성도 h가 낮은 사람(h<0.2)의 앱이 처치 쪽에서만 60% 확률로 로그를 못 남기는 버그(가상)."""
    rng = random.Random(seed)
    yt, yc = [], []
    for _ in range(n):
        h = rng.random()
        t = rng.random() < 0.5
        y = 4.0 + 16.0 * h + (SRM_EFFECT if t else 0.0) + normal(rng, 2.0)
        lost = bug and t and h < 0.2 and rng.random() < 0.6
        if lost:
            continue
        (yt if t else yc).append(y)
    chi2, p = srm_pvalue(len(yt), len(yc))
    return {"n_treat": len(yt), "n_control": len(yc), "chi2": chi2, "p": p,
            "estimate": mean(yt) - mean(yc), "true_effect": SRM_EFFECT}


# ---------- 4. 비순응: 배정과 실제 사용이 다르다 ----------
def noncompliance_demo(n=20000, seed=24):
    """배정받은 사람의 사용 확률 0.2+0.7h(열성도가 높을수록 쿠폰을 씀). 대조군은 쿠폰을 쓸 수 없다.
    효과 tau = 5 - 4h(열성도가 높을수록 효과가 작다)."""
    rng = random.Random(seed)
    z_y, c_y = [], []          # 배정군 전체, 대조군 전체
    users_y, users_tau = [], []
    for _ in range(n):
        h = rng.random()
        y0 = 4.0 + 16.0 * h + normal(rng, 2.0)
        tau = 5.0 - 4.0 * h
        z = rng.random() < 0.5
        if z:
            used = rng.random() < 0.2 + 0.7 * h
            y = y0 + (tau if used else 0.0)
            z_y.append(y)
            if used:
                users_y.append(y)
                users_tau.append(tau)
        else:
            c_y.append(y0)
    itt = mean(z_y) - mean(c_y)
    share = len(users_y) / len(z_y)
    return {"itt": itt, "share_used": share, "wald": itt / share,
            "per_protocol": mean(users_y) - mean(c_y),
            "true_complier_effect": mean(users_tau)}


if __name__ == "__main__":
    s = stratification_demo()
    print("1 층화: 단순 배정 평균 %.2f, SD %.2f / 층화 배정 평균 %.2f, SD %.2f (반복 %d, 진짜 %.1f)" % (
        s["simple_mean"], s["simple_sd"], s["strat_mean"], s["strat_sd"], s["reps"], EFFECT))
    c = cluster_demo()
    print("2 군집: 추정 평균 %.2f / 실제 SD %.2f / 독립 가정 SE %.2f / 95%% 구간 포함률 %.2f / 설계 효과 %.1f (rho %.3f)" % (
        c["mean_estimate"], c["actual_sd"], c["naive_se"], c["naive_coverage"], c["design_effect"], c["rho"]))
    for bug in (False, True):
        r = srm_demo(bug=bug)
        print("3 SRM(버그 %s): 처치 %d명 대조 %d명 chi2 %.2f p %.3g 추정 %.2f (진짜 %.1f)" % (
            "있음" if bug else "없음", r["n_treat"], r["n_control"], r["chi2"], r["p"], r["estimate"], r["true_effect"]))
    n = noncompliance_demo()
    print("4 비순응: ITT %.2f / 사용 비율 %.3f / 왈드 %.2f / 사용자-대조 비교 %.2f / 사용자의 진짜 효과 %.2f" % (
        n["itt"], n["share_used"], n["wald"], n["per_protocol"], n["true_complier_effect"]))
