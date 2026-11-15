"""13장: 재고 100개를 두 지역에 나눠 두기. 수요 시나리오에 과적합한 계획과 강건한 계획을 비교한다.

모든 수치는 학습용 가상값이다. 표준 라이브러리만 쓰고 시드를 고정해 결정적으로 돌아간다.

재는 것: 예측 수요 한 점에 맞춘 계획, 충격 3개 중 최악을 막는 강건 계획, 최대 후회를 줄이는 계획의 평균·하위 5%·최악 이익,
그리고 학습 시나리오가 적을 때 계획이 그 시나리오에 과적합되는 정도.
입력: 재고 100개, 지역별 마진 A 30·B 24천원, 예측 수요 (60, 50), 충격 폭 25, 실제 수요의 표준편차 20. 출력: A 지역 배치량 x_A와 이익(천원).
대응(README 기대 출력): 예측대로 x_A = 60(이익 2760), 강건 53, 후회 최소 56, n=3의 학습 안 이익 2489.5 대 새 시나리오 2394.2(차이 95.3),
충격 60이면 x_A = 50(집합 안 최악 1200, 집합 밖 1650), 손계산 3450-24x = 30x+600에서 x = 52.78.
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
    # 팔린 양 = min(둔 양, 수요). 수요를 넘겨 둔 몫과 남은 재고는 가치 0(모형의 단순화)
    d_a, d_b = demand
    return MARGIN["A"] * min(x_a, d_a) + MARGIN["B"] * min(stock - x_a, d_b)


def draw_scenarios(n, seed):
    """실제 수요 분포에서 n개 시나리오를 뽑는다. 음수 수요는 0으로 자른다."""
    # 지역 A, B 수요를 서로 독립인 정규분포(평균 = 예측, 표준편차 20)에서 뽑는다(가정). 정수로 반올림하고 음수는 0
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        d_a = max(0, round(rng.gauss(FORECAST[0], TRUE_SD)))
        d_b = max(0, round(rng.gauss(FORECAST[1], TRUE_SD)))
        out.append((d_a, d_b))
    return out


def plan_for_scenarios(scenarios):
    """시나리오 평균 이익이 가장 큰 x_a. 동점이면 작은 값."""
    # x_A를 0..100 전부 대입. 키의 -x는 동점일 때 작은 x를 고르게 하는 장치(결정적 선택)
    best = max(range(STOCK + 1), key=lambda x: (sum(profit(x, d) for d in scenarios) / len(scenarios), -x))
    return best


def shock_set(forecast=FORECAST, shock=SHOCK):
    """강건 계획이 대비할 수요: 예측 그대로, A만 낮아짐, B만 낮아짐 (한 번에 한 지역만 흔들린다)."""
    d_a, d_b = forecast
    return [(d_a, d_b), (max(0, d_a - shock), d_b), (d_a, max(0, d_b - shock))]  # 수요는 0 아래로 내려가지 않는다


def robust_plan(scenarios):
    """가장 나쁜 시나리오의 이익을 최대로 하는 x_a. 최악 이익이 같으면 예측 수요 이익이 큰 쪽, 그래도 같으면 작은 값."""
    # 키 순서: 1) 최악 이익 최대, 2) 동점이면 예측 수요 이익, 3) 그래도 같으면 작은 x
    return max(range(STOCK + 1), key=lambda x: (min(profit(x, d) for d in scenarios), profit(x, FORECAST), -x))


def regret_plan(scenarios):
    """시나리오마다 '알았더라면 냈을 최대 이익'과의 차이(후회)의 최댓값이 가장 작은 x_a."""
    # 후회 = (그 시나리오를 미리 알았다면 낼 최대 이익) - (이 계획의 이익). 시나리오별 최댓값을 먼저 구해 둔다
    best_each = [max(profit(x, d) for x in range(STOCK + 1)) for d in scenarios]
    def worst_regret(x):
        return max(b - profit(x, d) for b, d in zip(best_each, scenarios))
    return min(range(STOCK + 1), key=lambda x: (worst_regret(x), x))


def evaluate(x_a, scenarios):
    """평가용 시나리오에서 평균, 하위 5%, 최악 이익."""
    vals = sorted(profit(x_a, d) for d in scenarios)
    # 하위 5% 값: 오름차순 정렬에서 5% 위치의 값(보간 없음). 평균이 같아도 나쁜 꼬리가 다를 수 있다
    p5 = vals[int(0.05 * len(vals))]
    return {"mean": statistics.fmean(vals), "p5": p5, "min": vals[0]}


def overfit_gap(n_train, reps=200, test_n=2000, seed=0):
    """학습 시나리오 n_train개로 계획을 짜면, 같은 시나리오에서 본 이익이 새 시나리오에서 얼마나 줄어드는가."""
    # 학습용과 평가용 시나리오를 서로 다른 시드로 뽑는다. 평가 묶음 1개(2000건)를 reps회가 공유
    test = draw_scenarios(test_n, seed=10_000 + seed)
    gaps_in, gaps_out = [], []
    for r in range(reps):
        train = draw_scenarios(n_train, seed=seed * 1_000_003 + r)
        # 학습 시나리오 평균 이익이 가장 큰 x를 고르면 그 표본의 잡음까지 맞추게 된다(과적합)
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
    # 시나리오가 예측 한 점뿐이면: 마진 큰 A에 예측 수요 60을 먼저, 나머지 40을 B에
    x_fc = plan_for_scenarios([FORECAST])
    print("x_A =", x_fc, "예측 수요에서 이익 =", profit(x_fc, FORECAST))
    shocks = shock_set()
    x_rb = robust_plan(shocks)
    x_rg = regret_plan(shocks)
    print("강건(최악 이익 최대) x_A =", x_rb, " 후회 최소 x_A =", x_rg)
    # 계획을 짠 데이터와 다른 시드(777)의 시나리오 2000개로 평가: 세 계획 모두 평균은 비슷하고 하위 5%만 강건·후회 최소가 낫다
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
    # 주의: a, b는 위 for문 마지막 반복(n=200)의 값이 남은 것. 음수(-5.25)는 평가 묶음 하나의 표본 잡음으로 보인다(확인 필요)
    print("n=200 차이(반올림 전) =", round(a - b, 4))
    # A만 25 낮아질 때(수요 35, 50): 30*35 + 24*(100-x) = 3450-24x. B만 낮아질 때(60, 25): 30x + 24*25 = 30x+600. 두 직선의 교점 x = 2850/54
    print("손계산: 3450-24x = 30x+600 -> x =", 2850 / 54)


if __name__ == "__main__":
    main()
