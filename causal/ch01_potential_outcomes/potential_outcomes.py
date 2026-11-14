"""1장 실습: 잠재결과, ATE·ATT·ITT, 선택 편향. 모든 숫자는 가상(예시용)이다.

난수는 random.Random(seed).random()만 쓴다.
"""
import math
import random

# ---------- 신의 시점 표: 여덟 명의 Y0(쿠폰 없을 때), Y1(쿠폰 있을 때) 월 구매액(만원) ----------
CUSTOMERS = [
    # (이름, Y0, Y1, 쿠폰을 받았나)
    ("A", 20, 22, 1),
    ("B", 18, 20, 1),
    ("C", 16, 16, 1),
    ("D", 14, 14, 1),
    ("E", 6, 10, 0),
    ("F", 4, 8, 0),
    ("G", 4, 8, 0),
    ("H", 2, 6, 0),
]


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs)


def hand_table():
    ate = mean(y1 - y0 for _, y0, y1, _ in CUSTOMERS)
    att = mean(y1 - y0 for _, y0, y1, t in CUSTOMERS if t == 1)
    atc = mean(y1 - y0 for _, y0, y1, t in CUSTOMERS if t == 0)
    # 우리가 실제로 보는 값: 받은 사람은 Y1, 안 받은 사람은 Y0
    treated_obs = mean(y1 for _, y0, y1, t in CUSTOMERS if t == 1)
    control_obs = mean(y0 for _, y0, y1, t in CUSTOMERS if t == 0)
    naive = treated_obs - control_obs
    bias = mean(y0 for _, y0, y1, t in CUSTOMERS if t == 1) - mean(y0 for _, y0, y1, t in CUSTOMERS if t == 0)
    return {
        "ate": ate, "att": att, "atc": atc,
        "treated_obs": treated_obs, "control_obs": control_obs,
        "naive": naive, "selection_bias": bias,
    }


# ---------- 시뮬레이션: 열성도 h가 높은 사람이 쿠폰을 더 받는다 ----------
def normal(rng, sd=1.0):
    u1 = 1.0 - rng.random()
    u2 = rng.random()
    return sd * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def make_population(n=20000, seed=3):
    """각 사람의 (h, Y0, Y1). 효과 tau = 4 - 3h: 열성도가 낮은 사람에게 더 크다. 평균 효과는 2.5."""
    rng = random.Random(seed)
    pop = []
    for _ in range(n):
        h = rng.random()
        y0 = 4.0 + 16.0 * h + normal(rng, 2.0)
        y1 = y0 + 4.0 - 3.0 * h
        pop.append((h, y0, y1))
    return pop


def self_selected(pop, seed=4):
    """쿠폰을 직접 받아 가는 구조: 받을 확률이 0.1+0.8h."""
    rng = random.Random(seed)
    return [1 if rng.random() < 0.1 + 0.8 * h else 0 for h, _, _ in pop]


def randomized(pop, seed=5):
    rng = random.Random(seed)
    return [1 if rng.random() < 0.5 else 0 for _ in pop]


def estimates(pop, t):
    ate = mean(y1 - y0 for _, y0, y1 in pop)
    att = mean(y1 - y0 for (_, y0, y1), ti in zip(pop, t) if ti == 1)
    naive = (mean(y1 for (_, y0, y1), ti in zip(pop, t) if ti == 1)
             - mean(y0 for (_, y0, y1), ti in zip(pop, t) if ti == 0))
    bias = (mean(y0 for (_, y0, y1), ti in zip(pop, t) if ti == 1)
            - mean(y0 for (_, y0, y1), ti in zip(pop, t) if ti == 0))
    return {"ate": ate, "att": att, "naive": naive, "selection_bias": bias,
            "share_treated": mean(t)}


def itt_demo(pop, seed=6, takeup=0.6):
    """무작위로 쿠폰을 '배정'하지만 배정받은 사람 중 takeup 비율만 실제로 쓴다(열성도와 무관).
    ITT는 배정 기준 차이, 사용자 평균 효과는 ITT / takeup."""
    rng = random.Random(seed)
    z = [1 if rng.random() < 0.5 else 0 for _ in pop]
    used = [1 if (zi == 1 and rng.random() < takeup) else 0 for zi in z]
    y = [(y1 if u else y0) for (_, y0, y1), u in zip(pop, used)]
    itt = mean(yi for yi, zi in zip(y, z) if zi == 1) - mean(yi for yi, zi in zip(y, z) if zi == 0)
    return {"itt": itt, "takeup": mean(u for u, zi in zip(used, z) if zi == 1),
            "itt_over_takeup": itt / mean(u for u, zi in zip(used, z) if zi == 1)}


if __name__ == "__main__":
    h = hand_table()
    print("손계산 표: ATE %.1f / ATT %.1f / ATC %.1f / 단순 비교 %.1f = ATT %.1f + 선택 편향 %.1f" % (
        h["ate"], h["att"], h["atc"], h["naive"], h["att"], h["selection_bias"]))
    pop = make_population()
    for name, t in (("자기 선택", self_selected(pop)), ("무작위 배정", randomized(pop))):
        e = estimates(pop, t)
        print("%s: ATE %.2f / ATT %.2f / 단순 비교 %.2f / 선택 편향 %.2f / 받은 비율 %.3f" % (
            name, e["ate"], e["att"], e["naive"], e["selection_bias"], e["share_treated"]))
    i = itt_demo(pop)
    print("배정 대비 ITT %.2f / 실제 사용 비율 %.3f / ITT÷사용 비율 %.2f" % (i["itt"], i["takeup"], i["itt_over_takeup"]))
