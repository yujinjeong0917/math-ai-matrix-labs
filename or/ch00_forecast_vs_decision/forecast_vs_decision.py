"""0장: 예측 오차(MAE)가 작다고 결정이 좋은 것은 아니다. 모든 숫자는 가상 예시용.

재는 것: 지역별 수요 예측 두 개(P, Q)의 평균절대오차(MAE, Mean Absolute Error)와,
그 예측으로 재고 60켤레를 나눴을 때 실제 수요에서 얻는 이익.
입력: 모듈 상단 상수(마진, 실제 수요, 예측 P·Q, 재고). 출력: 예측별 MAE, 배분, 이익, 후회.
대응: README 기대 출력 "P: MAE=2.67 ... 이익=380,000원", "Q: MAE=3.33 ... 이익=420,000원"
(오차가 더 큰 Q가 이익은 더 크다), 그리고 재고 60 대 90에서 배분이 달라지는 두 줄.
"""
from fractions import Fraction

REGIONS = ("A", "B", "C")
# 오차를 일부러 한 지역에 몰아 넣은 구성(현실의 오차 분포가 아님). C는 마진이 가장 커서 C의 오차가 결정에 더 크게 번진다.
MARGIN = {"A": 4000, "B": 5000, "C": 9000}   # 켤레당 마진(원, 가상)
ACTUAL = {"A": 30, "B": 30, "C": 30}         # 실제 수요(켤레, 가상)
FORECAST_P = {"A": 30, "B": 30, "C": 38}     # 예측 P: 오차가 C에만 +8
FORECAST_Q = {"A": 40, "B": 30, "C": 30}     # 예측 Q: 오차가 A에만 +10
STOCK = 60


def mae(forecast, actual=ACTUAL):
    # 분수로 두어 8/3, 10/3을 부동소수점 오차 없이 비교한다(테스트의 Fraction(8, 3))
    return Fraction(sum(abs(forecast[r] - actual[r]) for r in REGIONS), len(REGIONS))


def allocate(demand, stock=STOCK, margin=MARGIN):
    """마진이 큰 지역부터 수요만큼 채운다(수요 상한이 있는 배분 문제의 최적해)."""
    # 마진 내림차순 정렬: 그리디가 최적인 이유는 지역마다 수요 상한만 있는 단순한 구조라서
    left, plan = stock, {r: 0 for r in REGIONS}
    for r in sorted(REGIONS, key=lambda r: -margin[r]):
        plan[r] = min(demand[r], left)
        left -= plan[r]
    return plan


def profit(plan, actual=ACTUAL, margin=MARGIN):
    # 실제 수요를 넘겨 배분한 물량은 팔리지 않으므로 min으로 잘라 이익에서 뺀다(부족분 손실은 모형 밖)
    return sum(margin[r] * min(plan[r], actual[r]) for r in REGIONS)


def brute_force_best(actual=ACTUAL, stock=STOCK):
    """allocate()를 쓰지 않는 두 번째 길: 가능한 모든 정수 배분을 센다."""
    # 지역 C는 남은 재고로 정해지므로 (a, b)만 돌리면 모든 합이 stock인 정수 배분을 빠짐없이 센다
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
    ORACLE = report()["oracle"][2]   # 실제 수요를 미리 알았을 때의 이익; 후회 = ORACLE - 이익
    for k, (m, plan, p) in report().items():
        print(f"{k}: MAE={float(m):.2f} 배분={plan} 이익={p:,}원 후회={ORACLE-p:,}원")
    print("전수 탐색 최대 이익:", f"{brute_force_best():,}")
    # 같은 예측 P, 재고만 90으로 늘리면 결정이 달라진다
    for s in (60, 90):
        print(f"재고 {s}: 배분={allocate(FORECAST_P, s)}")
