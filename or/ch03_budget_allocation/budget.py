"""3장: 광고 예산 두 채널 배분. 포화(수확 체감) 수익 함수의 해석해를 격자 탐색과 대조한다.

모형(가상 계수): 수익 = 12*ln(1+x_a) + 7*ln(1+3*x_b), x_a + x_b = budget(기본 6, 단위는 예산 단위).
입력: budget, 격자 간격 step. 출력: 채널별 배분과 수익. 학습용 가상 계수이며 실제 광고효과 추정치가 아니다.
이 장에는 기대 출력을 찍는 실행부가 없고, 숫자는 test_budget.py의 단언(x_a = 207/57, 최대 수익 약 33.042369)에 있다.
"""
import math

def revenue(x_a: float, budget: float = 6.0) -> float:
    # 예산을 모두 쓰는 배분만 다룬다(x_b는 나머지). 음수 배분은 불가능하므로 오류
    x_b = budget - x_a
    if x_a < 0 or x_b < 0:
        raise ValueError("allocation must be feasible")
    # log1p(t) = ln(1+t): t가 0 근처에서도 안정적이고 x=0일 때 수익 0. 채널 B는 효율 계수 3배 가정
    return 12 * math.log1p(x_a) + 7 * math.log1p(3 * x_b)

def analytic_solution(budget: float = 6.0) -> tuple[float, float]:
    # 1계 조건: 두 채널의 한계수익이 같다. d/dx_a: 12/(1+x_a), d/dx_b: 7*3/(1+3*x_b) = 21/(1+3*x_b)
    # 대입해 풀면 x_a = (36*budget - 9)/57. budget=6이면 207/57 (약 3.63)
    # 로그는 오목이라 이 정류점이 전역 최대. 단, x_a >= 0 조건은 따지지 않으므로 budget < 0.25에서는 쓸 수 없다(모형의 한계)
    x_a = (36 * budget - 9) / 57
    return x_a, budget - x_a

def grid_solution(budget: float = 6.0, step: float = 0.001) -> tuple[float, float, float]:
    # round: budget/step의 부동소수점 오차(예: 5.999999)로 마지막 격자점을 놓치지 않게 한다
    n = round(budget / step)
    candidates = ((i * step, revenue(i * step, budget)) for i in range(n + 1))
    x_a, value = max(candidates, key=lambda row: row[1])
    return x_a, budget - x_a, value
