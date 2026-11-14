"""13장: 재고 100개를 두 지역에 나눠 두기. 수요 시나리오에 과적합한 계획과 강건한 계획을 비교한다.

모든 수치는 학습용 가상값이다. 표준 라이브러리만 쓰고 시드를 고정해 결정적으로 돌아간다.
"""
import random
import statistics

STOCK = 100                  # 두 지역에 나눠 둘 운동화 재고(개)
MARGIN = {"A": 30, "B": 24}  # 한 켤레를 팔 때 남는 돈(천원). 남은 재고의 값은 0으로 둔다(가상).
FORECAST = (60, 50)          # 예측이 말하는 다음 주 수요(A, B)
SHOCK = 25                   # 한 지역 수요가 예측보다 이만큼 낮아질 수 있다고 보는 폭(가상)
TRUE_SD = 20                 # 실제 수요가 예측 주변에서 흔들리는 표준편차(가상)


def profit(x_a, demand, stock=STOCK):
    """A에 x_a개, B에 stock-x_a개를 둔 계획의 이익(천원). 수요보다 많이 둔 몫은 팔리지 않는다."""
    if not 0 <= x_a <= stock:
        raise ValueError("allocation must be feasible")
    d_a, d_b = demand
    return MARGIN["A"] * min(x_a, d_a) + MARGIN["B"] * min(stock - x_a, d_b)


def draw_scenarios(n, seed):
    """실제 수요 분포에서 n개 시나리오를 뽑는다. 음수 수요는 0으로 자른다."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        d_a = max(0, round(rng.gauss(FORECAST[0], TRUE_SD)))
        d_b = max(0, round(rng.gauss(FORECAST[1], TRUE_SD)))
        out.append((d_a, d_b))
    return out


def plan_for_scenarios(scenarios):
    """시나리오 평균 이익이 가장 큰 x_a. 동점이면 작은 값."""
    best = max(range(STOCK + 1), key=lambda x: (sum(profit(x, d) for d in scenarios) / len(scenarios), -x))
    return best


def shock_set(forecast=FORECAST, shock=SHOCK):
    """강건 계획이 대비할 수요: 예측 그대로, A만 낮아짐, B만 낮아짐 (한 번에 한 지역만 흔들린다)."""
    d_a, d_b = forecast
    return [(d_a, d_b), (max(0, d_a - shock), d_b), (d_a, max(0, d_b - shock))]  # 수요는 0 아래로 내려가지 않는다


def robust_plan(scenarios):
    """가장 나쁜 시나리오의 이익을 최대로 하는 x_a. 최악 이익이 같으면 예측 수요 이익이 큰 쪽, 그래도 같으면 작은 값."""
    return max(range(STOCK + 1), key=lambda x: (min(profit(x, d) for d in scenarios), profit(x, FORECAST), -x))


def regret_plan(scenarios):
    """시나리오마다 '알았더라면 냈을 최대 이익'과의 차이(후회)의 최댓값이 가장 작은 x_a."""
    best_each = [max(profit(x, d) for x in range(STOCK + 1)) for d in scenarios]
    def worst_regret(x):
        return max(b - profit(x, d) for b, d in zip(best_each, scenarios))
    return min(range(STOCK + 1), key=lambda x: (worst_regret(x), x))


def evaluate(x_a, scenarios):
    """평가용 시나리오에서 평균, 하위 5%, 최악 이익."""
    vals = sorted(profit(x_a, d) for d in scenarios)
    p5 = vals[int(0.05 * len(vals))]
    return {"mean": statistics.fmean(vals), "p5": p5, "min": vals[0]}


def overfit_gap(n_train, reps=200, test_n=2000, seed=0):
    """학습 시나리오 n_train개로 계획을 짜면, 같은 시나리오에서 본 이익이 새 시나리오에서 얼마나 줄어드는가."""
    test = draw_scenarios(test_n, seed=10_000 + seed)
    gaps_in, gaps_out = [], []
    for r in range(reps):
        train = draw_scenarios(n_train, seed=seed * 1_000_003 + r)
        x = plan_for_scenarios(train)
        gaps_in.append(statistics.fmean(profit(x, d) for d in train))
        gaps_out.append(statistics.fmean(profit(x, d) for d in test))
    return statistics.fmean(gaps_in), statistics.fmean(gaps_out)


def shock_sweep(shocks=(0, 10, 25, 40, 60), outside=(15, 50)):
    """충격 폭을 키울수록 강건 계획이 어디로 가는지, 그리고 집합 밖 수요(outside)에서의 이익."""
    rows = []
    for k in shocks:
        x = robust_plan(shock_set(shock=k))
        rows.append((k, x, profit(x, FORECAST), min(profit(x, d) for d in shock_set(shock=k)), profit(x, outside)))
    return rows


def main():
    print("== 예측 그대로(단일 시나리오) 계획 ==")
    x_fc = plan_for_scenarios([FORECAST])
    print("x_A =", x_fc, "예측 수요에서 이익 =", profit(x_fc, FORECAST))
    shocks = shock_set()
    x_rb = robust_plan(shocks)
    x_rg = regret_plan(shocks)
    print("강건(최악 이익 최대) x_A =", x_rb, " 후회 최소 x_A =", x_rg)
    test = draw_scenarios(2000, seed=777)
    print("== 새 시나리오 2000개에서 평가 ==")
    for name, x in [("예측 그대로", x_fc), ("강건", x_rb), ("후회 최소", x_rg)]:
        e = evaluate(x, test)
        print(f"{name:8s} x_A={x:3d} 예측수요이익={profit(x, FORECAST):5d} 충격 3개 중 최악={min(profit(x, d) for d in shocks):5d} 평균={e['mean']:.1f} 하위5%={e['p5']} 최악={e['min']}")
    print("== 학습 시나리오 개수별 과적합(200회 반복 평균) ==")
    for n in (3, 5, 20, 200):
        a, b = overfit_gap(n)
        print(f"n={n:4d} 학습 안 이익={a:.1f} 새 시나리오 이익={b:.1f} 차이={a-b:.1f}")
    t2 = draw_scenarios(2000, seed=10_000)
    print("== 5절 평가 묶음(seed 10000)에서 같은 계획의 평균:", {n: round(evaluate(x, t2)["mean"], 1) for n, x in [("예측60", x_fc), ("강건53", x_rb), ("후회56", x_rg)]})
    print("== 충격 폭별 강건 계획 (집합 밖 수요 A=15,B=50에서의 이익도 함께) ==")
    for k, x, fc, worst, out in shock_sweep():
        print(f"충격={k:2d} x_A={x:3d} 예측수요이익={fc} 집합 안 최악={worst} 집합 밖(15,50)={out}")
    print("n=200 차이(반올림 전) =", round(a - b, 4))
    print("손계산: 3450-24x = 30x+600 -> x =", 2850 / 54)


if __name__ == "__main__":
    main()
