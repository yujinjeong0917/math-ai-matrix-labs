import math

def revenue(x_a: float, budget: float = 6.0) -> float:
    x_b = budget - x_a
    if x_a < 0 or x_b < 0:
        raise ValueError("allocation must be feasible")
    return 12 * math.log1p(x_a) + 7 * math.log1p(3 * x_b)

def analytic_solution(budget: float = 6.0) -> tuple[float, float]:
    # 12/(1+x_a) = 21/(1+3*x_b), x_b = budget-x_a
    x_a = (36 * budget - 9) / 57
    return x_a, budget - x_a

def grid_solution(budget: float = 6.0, step: float = 0.001) -> tuple[float, float, float]:
    n = round(budget / step)
    candidates = ((i * step, revenue(i * step, budget)) for i in range(n + 1))
    x_a, value = max(candidates, key=lambda row: row[1])
    return x_a, budget - x_a, value
