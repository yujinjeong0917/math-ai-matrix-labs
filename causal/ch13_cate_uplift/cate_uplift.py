"""13장 실습: CATE, 업리프트, 사후 하위집단 탐색의 함정. 표준 라이브러리만 쓴다."""
import math
import random

COST = 1500  # 쿠폰 1장당 비용(원, 가상)
REGIONS = ("A", "B", "C")


def true_tau(new: int, heavy: int) -> float:
    """가상 세계의 진짜 쿠폰 효과(원). 신규 고객은 더 크고, 원래 많이 사던 고객은 더 작다."""
    return 1000 + 2500 * new - 1500 * heavy


def make_customers(n: int, seed: int):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        new = int(rng.random() < 0.4)
        heavy = int(rng.random() < 0.3)
        region = rng.choices(REGIONS, weights=[0.4, 0.35, 0.25])[0]
        t = int(rng.random() < 0.5)  # 무작위 배정
        base = 20000 + 8000 * heavy
        y = base + true_tau(new, heavy) * t + rng.gauss(0, 6000)
        rows.append({"new": new, "heavy": heavy, "region": region, "t": t, "y": y})
    return rows


def diff_means(rows):
    """처치-비교 평균 차이와 표준오차(웰치)."""
    a = [r["y"] for r in rows if r["t"] == 1]
    b = [r["y"] for r in rows if r["t"] == 0]
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    return ma - mb, math.sqrt(va / len(a) + vb / len(b))


def t_learner_cells(rows):
    """T-learner의 가장 작은 형태: 칸(신규 x 지역 x 다량구매)마다 처치군 평균 - 비교군 평균."""
    cells = {}
    for r in rows:
        cells.setdefault((r["new"], r["region"], r["heavy"]), []).append(r)
    return {k: diff_means(v)[0] for k, v in sorted(cells.items())}


def cell_stats(rows):
    """칸별 (인원, 추정 효과, 표준오차)."""
    cells = {}
    for r in rows:
        cells.setdefault((r["new"], r["region"], r["heavy"]), []).append(r)
    return {k: (len(v),) + diff_means(v) for k, v in sorted(cells.items())}


def policy_gain(cell_uplift, rows):
    """추정 업리프트가 비용보다 큰 칸에만 쿠폰을 줄 때, 진짜 효과로 센 순이익(원/명)."""
    total = 0.0
    for r in rows:
        est = cell_uplift.get((r["new"], r["region"], r["heavy"]), 0.0)
        if est > COST:
            total += true_tau(r["new"], r["heavy"]) - COST
    return total / len(rows)


def treat_all_gain(rows):
    return sum(true_tau(r["new"], r["heavy"]) - COST for r in rows) / len(rows)


def normal_two_sided_p(z):
    return math.erfc(abs(z) / math.sqrt(2))


def z_for_two_sided(alpha):
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if normal_two_sided_p(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def null_world(n: int, n_feat: int, seed: int):
    """처치 효과가 모든 사람에게 0인 무작위 실험. 하위집단 후보 특성은 동전 던지기."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        feats = [int(rng.random() < 0.5) for _ in range(n_feat)]
        rows.append({"f": feats, "t": int(rng.random() < 0.5), "y": 20000 + rng.gauss(0, 6000)})
    return rows


def best_subgroup(rows, n_feat):
    """모든 (특성, 값) 하위집단에서 |t|가 가장 큰 것을 찾는다."""
    best = (0.0, None, 0.0)
    for j in range(n_feat):
        for v in (0, 1):
            sub = [{"t": r["t"], "y": r["y"]} for r in rows if r["f"][j] == v]
            est, se = diff_means(sub)
            if se == se and se > 0 and abs(est / se) > abs(best[0]):
                best = (est / se, (j, v), est)
    return best


def subgroup_experiment(reps=200, n=2000, n_feat=20, seed0=1000):
    z05 = z_for_two_sided(0.05)
    z_bonf = z_for_two_sided(0.05 / (2 * n_feat))
    hit = hit_bonf = hold_hit = 0
    max_ts, hold_est = [], []
    for k in range(reps):
        rows = null_world(n, n_feat, seed0 + k)
        half = n // 2
        train, hold = rows[:half], rows[half:]
        tstat, key, _ = best_subgroup(train, n_feat)
        max_ts.append(abs(tstat))
        hit += abs(tstat) > z05
        hit_bonf += abs(tstat) > z_bonf
        j, v = key
        sub = [{"t": r["t"], "y": r["y"]} for r in hold if r["f"][j] == v]
        est, se = diff_means(sub)
        hold_est.append(est)
        hold_hit += abs(est / se) > z05
    return {
        "reps": reps, "z05": z05, "z_bonf": z_bonf,
        "share_best_sig": hit / reps, "share_best_sig_bonf": hit_bonf / reps,
        "mean_max_abs_t": sum(max_ts) / reps,
        "holdout_mean_est": sum(hold_est) / reps,
        "holdout_share_sig": hold_hit / reps,
    }


def main():
    rows = make_customers(20000, seed=2)
    half = len(rows) // 2
    train, hold = rows[:half], rows[half:]
    ate, se = diff_means(rows)
    print(f"전체 평균효과 추정 {ate:.0f}원 (표준오차 {se:.0f}), 진짜 평균 "
          f"{sum(true_tau(r['new'], r['heavy']) for r in rows) / len(rows):.0f}원")
    for name, f in (("신규", lambda r: r["new"] == 1), ("기존", lambda r: r["new"] == 0)):
        e, s = diff_means([r for r in rows if f(r)])
        print(f"사전에 정한 하위집단 {name}: {e:.0f}원 (표준오차 {s:.0f})")
    cells = t_learner_cells(train)
    print("칸별 추정 업리프트(학습 절반), 칸=(신규,지역,다량구매):")
    stats = cell_stats(train)
    for k, v in cells.items():
        n_c, _, se_c = stats[k]
        print(f"  {k}: 추정 {v:.0f}원 (표준오차 {se_c:.0f}, {n_c}명) / 진짜 {true_tau(k[0], k[2]):.0f}원")
    print(f"정책 순이익(보류 절반, 원/명): 업리프트 기준 {policy_gain(cells, hold):.0f}, "
          f"모두에게 {treat_all_gain(hold):.0f}, 아무에게도 0")
    res = subgroup_experiment()
    print("효과가 0인 세계에서 하위집단 탐색:")
    for k, v in res.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
