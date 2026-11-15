"""15장: 예측 -> 증분효과 -> 행동 -> 측정. 관찰 ROAS 순서와 증분 ROAS 순서가 갈리는 가상 시장.

모든 수치는 학습용 가상값이다. 표준 라이브러리만 쓰고 시드를 고정해 결정적으로 돌아간다.
돈 단위는 천원, 구매 한 건 매출은 30천원, 노출 한 번 비용은 1천원이다.

재는 것: 평소 운영에서 본 구매율과 ROAS(Return On Ad Spend, 광고비 대비 매출)가 광고 효과(증분)와 얼마나 다른지.
채널 A는 원래 잘 사는 사람에게 노출되어 관찰 ROAS가 높지만 참 효과(lift)는 작고, 채널 B는 반대다.
입력: 채널별 인원·구매 확률·참 lift, 예산 50,000천원. 출력: 관찰 ROAS, 무작위 대조 실험의 증분 ROAS(95% 구간), 예산 배분 계획의 참 증분 매출.
대응(README 기대 출력): A 노출 23503명 구매율 0.3392 관찰 ROAS 10.18, 단순차이 0.0995 대 참값 0.02, 실험 lift 0.0257, 증분 ROAS 0.77 [0.43, 1.11],
관찰 ROAS 순서 계획의 참 증분 매출 48000천원(증분 ROAS 0.96), 증분 ROAS 순서 계획 75000천원(1.50), 계획 때 예상 76173 / 측정 78089 [68914, 87264] / 참값 75000.
참 lift는 시뮬레이션이라서 아는 값이며 실제 데이터에서는 알 수 없다.
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
    # 사람마다 광고가 없어도 살 확률 p0를 base ~ base+spread 균등분포로 둔다(모형의 선택). 시드 고정으로 같은 집단이 재현된다
    rng = random.Random(seed)
    return [cfg["base"] + cfg["spread"] * rng.random() for _ in range(cfg["n"])]


def observe(name, seed):
    """평소 운영: 원래 잘 사는 사람에게 광고가 더 자주 간다(노출 확률 = min(0.95, 0.2 + 2*p0))."""
    # 선택 편향: 원래 잘 사는 사람(p0 큼)일수록 광고가 더 자주 간다. 그래서 노출군과 미노출군은 처음부터 다른 사람들이다
    p0 = make_pool(name, seed)
    rng = random.Random(seed + 1)
    lift = POOLS[name]["lift"]
    exp_n = exp_buy = un_n = un_buy = 0
    for p in p0:
        exposed = rng.random() < min(0.95, 0.2 + 2 * p)
        # 구매 확률 = 기본 확률 + (노출되면) 참 lift
        buy = rng.random() < p + (lift if exposed else 0.0)
        if exposed:
            exp_n += 1; exp_buy += buy
        else:
            un_n += 1; un_buy += buy
    return {"exp_n": exp_n, "exp_rate": exp_buy / exp_n, "un_rate": un_buy / un_n,
            # 관찰 ROAS = 노출된 사람의 전체 구매 매출 / 노출 비용. 광고 없이도 샀을 구매까지 포함해 부풀려진다
            "obs_roas": exp_buy * REVENUE / (exp_n * COST),
            # 단순 차이(노출군 - 미노출군 구매율)는 선택 편향이 섞여 참 lift(A는 0.02)보다 훨씬 크다(0.0995)
            "naive_lift": exp_buy / exp_n - un_buy / un_n}


def experiment(name, seed, holdout=0.3):
    """무작위로 holdout 비율만큼은 광고를 끊고 나머지는 노출한다. 증분 ROAS를 구간과 함께 돌려준다."""
    # 무작위 대조: 노출 여부를 p0와 무관하게 동전 던지기로 정하므로 두 집단이 평균적으로 같다 -> 차이가 인과 효과
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
    # 두 비율의 차이의 표준오차(정규 근사). 95% 구간 = 차이 +- 1.96 x se
    se = math.sqrt(pt * (1 - pt) / t_n + pc * (1 - pc) / c_n)
    k = REVENUE / COST   # 증분 ROAS = lift x 매출 / 비용. 1 미만이면 광고비가 증분 매출보다 크다
    return {"lift": d, "se": se, "lo": d - Z95 * se, "hi": d + Z95 * se,
            "iroas": d * k, "iroas_lo": (d - Z95 * se) * k, "iroas_hi": (d + Z95 * se) * k,
            "t_n": t_n, "c_n": c_n}


def plan_by(order, score, budget=BUDGET, min_score=None):
    """점수가 높은 채널부터 대상 집단을 노출 한 번당 비용으로 채운다. min_score 미만이면 쓰지 않는다."""
    # 점수가 높은 채널부터 예산을 채운다. min_score 미만 채널은 아예 집행하지 않는다(증분 ROAS 1 미만 보류)
    plan, left = {}, budget
    for name in sorted(order, key=lambda k: -score[k]):
        if min_score is not None and score[name] < min_score:
            plan[name] = 0
            continue
        # 노출 수 = 채널 인원과 남은 예산(노출당 1천원) 중 작은 값. B는 인원 70000 > 예산 50000이라 50000명
        n = int(min(POOLS[name]["n"], left // COST))
        plan[name] = n
        left -= n * COST
    return plan


def true_incremental(plan):
    """참값 기준 증분 매출(천원). 시뮬레이션이라 알 수 있다."""
    # 노출 수 x 참 lift x 매출. 예: A 30000 x 0.02 x 30 = 18000, B 20000 x 0.05 x 30 = 30000 -> 48000
    return sum(n * POOLS[k]["lift"] * REVENUE for k, n in plan.items())


def spend(plan):
    return sum(plan.values()) * COST


def predicted_vs_actual(name, seed_fit=101, seed_new=202):
    """예측: 평소 운영에서 노출된 사람의 구매율. 이번 달 같은 방식 운영에서 얼마나 맞는가."""
    # 예측이 정확한 것(오차 0.01 미만)과 광고 효과를 아는 것은 별개: 예측은 구매율만 맞출 뿐 lift는 말해주지 않는다
    fit = observe(name, seed_fit)
    new = observe(name, seed_new)
    return fit["exp_rate"], new["exp_rate"]


def measure_plan(plan, seed, holdout=0.1):
    """계획대로 집행하면서 노출 대상 중 holdout 비율을 무작위로 끊어 증분 매출을 잰다."""
    total, var = 0.0, 0.0
    for name, n in plan.items():
        if n == 0:
            continue
        # 이번 달 집단은 시드 404로 새로 뽑고, 집행 대상은 앞에서 n명(채널 B는 50000명)
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
        # 노출 n명 규모로 확대한 증분 매출 추정. 분산은 채널 간 독립이라 더한다
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
    # 계획 때 예상 = 2절 실험의 lift 추정치 x 집행 수 x 매출. 다음 달 측정과 참값을 같이 놓고 구간이 참값을 덮는지 본다
    pred = sum(n * ex[k]["lift"] * REVENUE for k, n in plan_inc.items())
    est, lo, hi = measure_plan(plan_inc, 404)
    # 아래 첫 숫자는 계획이 B 50000명뿐이라는 전제의 손계산(10% 대조군이 놓친 참 증분 매출)
    print(f"끊은 10% 몫의 참 증분 매출(포기한 몫) {int(50000 * 0.10) * POOLS['B']['lift'] * REVENUE:.0f}천원 / 측정 구간 하한의 증분 ROAS {lo / BUDGET:.2f}")
    print(f"계획 때 예상 {pred:.0f}천원 / 다음 달 측정 {est:.0f}천원 [{lo:.0f},{hi:.0f}] / 참값 {true_incremental(plan_inc):.0f}천원")


if __name__ == "__main__":
    main()
