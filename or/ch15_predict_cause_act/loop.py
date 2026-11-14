"""15장: 예측 -> 증분효과 -> 행동 -> 측정. 관찰 ROAS 순서와 증분 ROAS 순서가 갈리는 가상 시장.

모든 수치는 학습용 가상값이다. 표준 라이브러리만 쓰고 시드를 고정해 결정적으로 돌아간다.
돈 단위는 천원, 구매 한 건 매출은 30천원, 노출 한 번 비용은 1천원이다.
"""
import math
import random

REVENUE = 30.0     # 구매 한 건 매출(천원)
COST = 1.0         # 노출 한 번 비용(천원)
Z95 = 1.96
# 채널별 대상 집단. n=사람 수, 구매 확률 p0 = base + spread * U(0,1) (광고 없이도 사는 정도), lift = 광고가 확률에 더하는 참값(시뮬레이션만 안다)
POOLS = {
    "A": {"n": 30000, "base": 0.15, "spread": 0.30, "lift": 0.02},   # 장바구니 이탈자 재노출. 원래 잘 산다.
    "B": {"n": 70000, "base": 0.00, "spread": 0.08, "lift": 0.05},   # 처음 보는 사람 노출. 원래는 잘 안 산다.
}
BUDGET = 50000.0   # 다음 달 광고 예산(천원)


def make_pool(name, seed):
    cfg = POOLS[name]
    rng = random.Random(seed)
    return [cfg["base"] + cfg["spread"] * rng.random() for _ in range(cfg["n"])]


def observe(name, seed):
    """평소 운영: 원래 잘 사는 사람에게 광고가 더 자주 간다(노출 확률 = min(0.95, 0.2 + 2*p0))."""
    p0 = make_pool(name, seed)
    rng = random.Random(seed + 1)
    lift = POOLS[name]["lift"]
    exp_n = exp_buy = un_n = un_buy = 0
    for p in p0:
        exposed = rng.random() < min(0.95, 0.2 + 2 * p)
        buy = rng.random() < p + (lift if exposed else 0.0)
        if exposed:
            exp_n += 1; exp_buy += buy
        else:
            un_n += 1; un_buy += buy
    return {"exp_n": exp_n, "exp_rate": exp_buy / exp_n, "un_rate": un_buy / un_n,
            "obs_roas": exp_buy * REVENUE / (exp_n * COST),
            "naive_lift": exp_buy / exp_n - un_buy / un_n}


def experiment(name, seed, holdout=0.3):
    """무작위로 holdout 비율만큼은 광고를 끊고 나머지는 노출한다. 증분 ROAS를 구간과 함께 돌려준다."""
    p0 = make_pool(name, seed)
    rng = random.Random(seed + 2)
    lift = POOLS[name]["lift"]
    t_n = t_buy = c_n = c_buy = 0
    for p in p0:
        treated = rng.random() >= holdout
        buy = rng.random() < p + (lift if treated else 0.0)
        if treated:
            t_n += 1; t_buy += buy
        else:
            c_n += 1; c_buy += buy
    pt, pc = t_buy / t_n, c_buy / c_n
    d = pt - pc
    se = math.sqrt(pt * (1 - pt) / t_n + pc * (1 - pc) / c_n)
    k = REVENUE / COST
    return {"lift": d, "se": se, "lo": d - Z95 * se, "hi": d + Z95 * se,
            "iroas": d * k, "iroas_lo": (d - Z95 * se) * k, "iroas_hi": (d + Z95 * se) * k,
            "t_n": t_n, "c_n": c_n}


def plan_by(order, score, budget=BUDGET, min_score=None):
    """점수가 높은 채널부터 대상 집단을 노출 한 번당 비용으로 채운다. min_score 미만이면 쓰지 않는다."""
    plan, left = {}, budget
    for name in sorted(order, key=lambda k: -score[k]):
        if min_score is not None and score[name] < min_score:
            plan[name] = 0
            continue
        n = int(min(POOLS[name]["n"], left // COST))
        plan[name] = n
        left -= n * COST
    return plan


def true_incremental(plan):
    """참값 기준 증분 매출(천원). 시뮬레이션이라 알 수 있다."""
    return sum(n * POOLS[k]["lift"] * REVENUE for k, n in plan.items())


def spend(plan):
    return sum(plan.values()) * COST


def predicted_vs_actual(name, seed_fit=101, seed_new=202):
    """예측: 평소 운영에서 노출된 사람의 구매율. 이번 달 같은 방식 운영에서 얼마나 맞는가."""
    fit = observe(name, seed_fit)
    new = observe(name, seed_new)
    return fit["exp_rate"], new["exp_rate"]


def measure_plan(plan, seed, holdout=0.1):
    """계획대로 집행하면서 노출 대상 중 holdout 비율을 무작위로 끊어 증분 매출을 잰다."""
    total, var = 0.0, 0.0
    for name, n in plan.items():
        if n == 0:
            continue
        p0 = make_pool(name, seed)[:n]
        rng = random.Random(seed + 3)
        lift = POOLS[name]["lift"]
        t_n = t_buy = c_n = c_buy = 0
        for p in p0:
            treated = rng.random() >= holdout
            buy = rng.random() < p + (lift if treated else 0.0)
            if treated:
                t_n += 1; t_buy += buy
            else:
                c_n += 1; c_buy += buy
        pt, pc = t_buy / t_n, c_buy / c_n
        total += (pt - pc) * n * REVENUE
        var += (pt * (1 - pt) / t_n + pc * (1 - pc) / c_n) * (n * REVENUE) ** 2
    se = math.sqrt(var)
    return total, total - Z95 * se, total + Z95 * se


def main():
    print("== 1. 예측: 평소 운영에서 노출된 사람의 구매율 ==")
    obs = {}
    for k in POOLS:
        o = observe(k, 101)
        obs[k] = o
        fit, new = predicted_vs_actual(k)
        print(f"{k}: 노출 {o['exp_n']}명 구매율 {o['exp_rate']:.4f} 미노출 {o['un_rate']:.4f} 관찰ROAS {o['obs_roas']:.2f} | 예측 {fit:.4f} 이번 달 {new:.4f} 오차 {abs(fit-new):.4f}")
    print("== 2. 증분효과: 단순 노출-미노출 차이와 무작위 대조 ==")
    ex = {}
    for k in POOLS:
        e = experiment(k, 303)
        ex[k] = e
        print(f"   무작위 대조 인원: 노출 {e['t_n']}명 / 광고를 끊은 {e['c_n']}명 (채널 전체 {POOLS[k]['n']}명 중 30%를 끊음), 구간 반폭 {(e['hi'] - e['lo']) / 2:.4f}")
        print(f"{k}: 단순차이 {obs[k]['naive_lift']:.4f} 참값 {POOLS[k]['lift']:.2f} 실험 lift {e['lift']:.4f} [{e['lo']:.4f},{e['hi']:.4f}] 증분ROAS {e['iroas']:.2f} [{e['iroas_lo']:.2f},{e['iroas_hi']:.2f}]")
    print("== 3. 행동: 예산 %d천원 ==" % BUDGET)
    obs_score = {k: obs[k]["obs_roas"] for k in POOLS}
    inc_score = {k: ex[k]["iroas"] for k in POOLS}
    plan_obs = plan_by(POOLS, obs_score)
    plan_inc = plan_by(POOLS, inc_score, min_score=1.0)
    for name, plan in (("관찰ROAS 순서", plan_obs), ("증분ROAS 순서(1 미만은 보류)", plan_inc)):
        t = true_incremental(plan)
        print(f"{name}: {plan} 집행 {spend(plan):.0f}천원 참 증분매출 {t:.0f}천원 증분ROAS {t / spend(plan):.2f}")
    print("== 4. 측정: 다음 달에 계획을 집행하며 10% 무작위 대조 ==")
    pred = sum(n * ex[k]["lift"] * REVENUE for k, n in plan_inc.items())
    est, lo, hi = measure_plan(plan_inc, 404)
    print(f"끊은 10% 몫의 참 증분 매출(포기한 몫) {int(50000 * 0.10) * POOLS['B']['lift'] * REVENUE:.0f}천원 / 측정 구간 하한의 증분 ROAS {lo / BUDGET:.2f}")
    print(f"계획 때 예상 {pred:.0f}천원 / 다음 달 측정 {est:.0f}천원 [{lo:.0f},{hi:.0f}] / 참값 {true_incremental(plan_inc):.0f}천원")


if __name__ == "__main__":
    main()
