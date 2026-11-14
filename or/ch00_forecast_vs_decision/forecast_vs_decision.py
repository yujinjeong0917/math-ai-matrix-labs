"""0장: 예측 오차(MAE)가 작다고 결정이 좋은 것은 아니다. 모든 숫자는 가상 예시용."""
from fractions import Fraction

REGIONS = ("A", "B", "C")
MARGIN = {"A": 4000, "B": 5000, "C": 9000}   # 켤레당 마진(원, 가상)
ACTUAL = {"A": 30, "B": 30, "C": 30}         # 실제 수요(켤레, 가상)
FORECAST_P = {"A": 30, "B": 30, "C": 38}     # 예측 P: 오차가 C에만 +8
FORECAST_Q = {"A": 40, "B": 30, "C": 30}     # 예측 Q: 오차가 A에만 +10
STOCK = 60


def mae(forecast, actual=ACTUAL):
    return Fraction(sum(abs(forecast[r] - actual[r]) for r in REGIONS), len(REGIONS))


def allocate(demand, stock=STOCK, margin=MARGIN):
    """마진이 큰 지역부터 수요만큼 채운다(수요 상한이 있는 배분 문제의 최적해)."""
    left, plan = stock, {r: 0 for r in REGIONS}
    for r in sorted(REGIONS, key=lambda r: -margin[r]):
        plan[r] = min(demand[r], left)
        left -= plan[r]
    return plan


def profit(plan, actual=ACTUAL, margin=MARGIN):
    return sum(margin[r] * min(plan[r], actual[r]) for r in REGIONS)


def brute_force_best(actual=ACTUAL, stock=STOCK):
    """allocate()를 쓰지 않는 두 번째 길: 가능한 모든 정수 배분을 센다."""
    best = 0
    for a in range(stock + 1):
        for b in range(stock - a + 1):
            c = stock - a - b
            best = max(best, profit({"A": a, "B": b, "C": c}, actual))
    return best


def report():
    out = {}
    for name, f in (("P", FORECAST_P), ("Q", FORECAST_Q)):
        plan = allocate(f)
        out[name] = (mae(f), plan, profit(plan))
    oracle = allocate(ACTUAL)
    out["oracle"] = (Fraction(0), oracle, profit(oracle))
    return out


if __name__ == "__main__":
    ORACLE = report()["oracle"][2]
    for k, (m, plan, p) in report().items():
        print(f"{k}: MAE={float(m):.2f} 배분={plan} 이익={p:,}원 후회={ORACLE-p:,}원")
    print("전수 탐색 최대 이익:", f"{brute_force_best():,}")
    # 같은 예측 P, 재고만 90으로 늘리면 결정이 달라진다
    for s in (60, 90):
        print(f"재고 {s}: 배분={allocate(FORECAST_P, s)}")
