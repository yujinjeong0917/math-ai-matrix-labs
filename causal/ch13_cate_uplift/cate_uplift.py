"""13장 실습: CATE, 업리프트, 사후 하위집단 탐색의 함정. 표준 라이브러리만 쓴다.

무엇을 재는가: CATE(Conditional Average Treatment Effect, 특성이 같은 사람들의 평균 효과)를 칸(신규 x 지역 x 다량구매)별로 추정하고,
추정 업리프트(쿠폰이 만드는 추가 매출)가 비용보다 큰 칸에만 쿠폰을 주는 정책이 "모두에게 주기"보다 순이익이 큰지 본다.
또 효과가 0인 세계에서 하위집단을 마음껏 탐색하면 우연히 "유의한" 집단이 얼마나 자주 나오는지, 보류 데이터와 다중비교 보정이 이를 어떻게 막는지 본다.
입력: make_customers(n, seed)의 행(new: 신규, heavy: 다량구매, region, t: 무작위 배정, y: 월 매출 원, 가상). 출력: 칸별 추정, 정책 순이익(원/명), 탐색 통계.
테스트의 기대값(정책 순이익 우위 300원 초과, 최선 하위집단 유의 비율 0.4 초과 등)이다(장 본문 숫자와의 대응은 README로 확인 필요). 모든 금액은 가상이다.
"""
import math
import random

COST = 1500  # 쿠폰 1장당 비용(원, 가상). 추정 업리프트가 이보다 커야 줄 가치가 있다
REGIONS = ("A", "B", "C")


def true_tau(new: int, heavy: int) -> float:
    """가상 세계의 진짜 쿠폰 효과(원). 신규 고객은 더 크고, 원래 많이 사던 고객은 더 작다."""
    # 칸별 진짜 효과: 신규+비다량 3500, 신규+다량 2000, 기존+비다량 1000, 기존+다량 -500. 비용 1500 기준으로 앞의 둘만 줄 가치가 있다
    return 1000 + 2500 * new - 1500 * heavy


def make_customers(n: int, seed: int):
    """고객 n명. 신규 40%, 다량구매 30%, 지역 A/B/C = 0.40/0.35/0.25. 지역은 효과와 무관하다(잡음 칸을 만든다)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        new = int(rng.random() < 0.4)
        heavy = int(rng.random() < 0.3)
        region = rng.choices(REGIONS, weights=[0.4, 0.35, 0.25])[0]
        # 쿠폰 배정은 동전(무작위)이라 칸 안에서 처치군-비교군 차이가 그 칸의 효과를 불편 추정한다
        t = int(rng.random() < 0.5)  # 무작위 배정
        base = 20000 + 8000 * heavy
        y = base + true_tau(new, heavy) * t + rng.gauss(0, 6000)  # 잡음 SD 6000은 효과(1000~3500)보다 커서 작은 칸은 추정이 불안정
        rows.append({"new": new, "heavy": heavy, "region": region, "t": t, "y": y})
    return rows


def diff_means(rows):
    """처치-비교 평균 차이와 표준오차(웰치)."""
    a = [r["y"] for r in rows if r["t"] == 1]
    b = [r["y"] for r in rows if r["t"] == 0]
    if len(a) < 2 or len(b) < 2:  # 표준오차(분모 n - 1)를 못 구하는 작은 칸은 nan
        return float("nan"), float("nan")
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    # 웰치 표준오차: 두 집단 분산이 달라도 쓸 수 있는 형태 sqrt(va/na + vb/nb)
    return ma - mb, math.sqrt(va / len(a) + vb / len(b))


def t_learner_cells(rows):
    """T-learner의 가장 작은 형태: 칸(신규 x 지역 x 다량구매)마다 처치군 평균 - 비교군 평균.
    T-learner는 처치군과 비교군의 결과모형을 따로 맞춰 차이를 내는 방법이고, 여기서는 모형 대신 칸 평균을 쓴다."""
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
    """추정 업리프트가 비용보다 큰 칸에만 쿠폰을 줄 때, 진짜 효과로 센 순이익(원/명).
    학습 절반으로 칸 추정을 만들고 보류 절반에 적용하는 식으로 쓴다. 추정이 아니라 true_tau로 이익을 세는 것은 시뮬레이션이라 가능한 평가다."""
    total = 0.0
    for r in rows:
        # 학습에 없는 칸은 0(안 줌)으로 둔다
        est = cell_uplift.get((r["new"], r["region"], r["heavy"]), 0.0)
        if est > COST:
            total += true_tau(r["new"], r["heavy"]) - COST
    return total / len(rows)


def treat_all_gain(rows):
    """모두에게 줄 때의 순이익(원/명): 진짜 효과 평균 - 비용."""
    return sum(true_tau(r["new"], r["heavy"]) - COST for r in rows) / len(rows)


def normal_two_sided_p(z):
    """표준정규에서 양측 p값 2(1 - Phi(|z|)) = erfc(|z| / sqrt(2))."""
    return math.erfc(abs(z) / math.sqrt(2))


def z_for_two_sided(alpha):
    """양측 p값이 alpha가 되는 임계 z를 이분법으로 찾는다(0.05 -> 1.96). 표준 라이브러리에 역누적분포가 없어 직접 푼다."""
    lo, hi = 0.0, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if normal_two_sided_p(mid) > alpha:  # p가 아직 크면 z를 더 키워야 한다
            lo = mid
        else:
            hi = mid
    return hi


def null_world(n: int, n_feat: int, seed: int):
    """처치 효과가 모든 사람에게 0인 무작위 실험. 하위집단 후보 특성은 동전 던지기.
    여기서 "유의한 하위집단"이 나오면 모두 우연이다(귀무가설이 참인 세계)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        feats = [int(rng.random() < 0.5) for _ in range(n_feat)]
        rows.append({"f": feats, "t": int(rng.random() < 0.5), "y": 20000 + rng.gauss(0, 6000)})
    return rows


def best_subgroup(rows, n_feat):
    """모든 (특성, 값) 하위집단에서 |t|가 가장 큰 것을 찾는다. 반환: (t값, (특성 번호, 값), 효과 추정).
    특성 n_feat개 x 값 2개 = 검정 2 n_feat번을 동시에 하는 셈이라 다중비교 문제가 생긴다."""
    best = (0.0, None, 0.0)
    for j in range(n_feat):
        for v in (0, 1):
            sub = [{"t": r["t"], "y": r["y"]} for r in rows if r["f"][j] == v]
            est, se = diff_means(sub)
            # se == se는 NaN 검사(NaN은 자기 자신과 같지 않다). 칸이 너무 작아 추정 불가인 경우를 건너뛴다
            if se == se and se > 0 and abs(est / se) > abs(best[0]):
                best = (est / se, (j, v), est)
    return best


def subgroup_experiment(reps=200, n=2000, n_feat=20, seed0=1000):
    """효과 0인 세계를 reps번 만들어, 학습 절반에서 고른 최선 하위집단이 (a) 보정 없이, (b) 본페로니 보정 후, (c) 보류 절반에서 유의한 비율을 센다."""
    z05 = z_for_two_sided(0.05)
    # 본페로니(Bonferroni) 보정: 유의수준을 검정 수 2 n_feat로 나눈다(보수적이지만 단순하다)
    z_bonf = z_for_two_sided(0.05 / (2 * n_feat))
    hit = hit_bonf = hold_hit = 0
    max_ts, hold_est = [], []
    for k in range(reps):
        rows = null_world(n, n_feat, seed0 + k)
        half = n // 2
        train, hold = rows[:half], rows[half:]  # 고객은 이미 무작위 순서라 앞뒤 절반이 곧 무작위 분할
        tstat, key, _ = best_subgroup(train, n_feat)
        max_ts.append(abs(tstat))
        hit += abs(tstat) > z05  # bool을 정수 1/0으로 더한다
        hit_bonf += abs(tstat) > z_bonf
        j, v = key  # 학습에서 고른 하위집단을 보류 데이터에서 "한 번만" 검정한다(탐색 없음)
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
    # 순서: 전체 평균 -> 사전 지정 하위집단 -> 칸별 업리프트(학습 절반) -> 정책 평가(보류 절반) -> 효과 0 세계의 함정
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
