"""1장 실습: 잠재결과, ATE·ATT·ITT, 선택 편향. 모든 숫자는 가상(예시용)이다.

난수는 random.Random(seed).random()만 쓴다.

무엇을 재는가: 같은 사람의 Y0(처치 없을 때)와 Y1(처치 있을 때)을 모두 아는 "신의 시점"에서
ATE(전체 평균 효과), ATT(처치받은 사람의 평균 효과), ATC(안 받은 사람의 평균 효과)를 구하고,
실제로 관찰되는 값만으로 낸 단순 비교가 ATT + 선택 편향으로 쪼개지는 것을 확인한다.
입력: 손계산 표 8명(CUSTOMERS), 시뮬레이션 인구 make_population(n, seed). 단위는 월 구매액(만원).
출력: ate/att/naive/selection_bias 등을 담은 dict. 테스트의 기대값(2.5, 1.0, 4.0, 14.0, 13.0 등)이 장 본문의 손계산 표와 대응한다.
ITT = Intent-To-Treat, 실제 사용 여부와 무관하게 "배정"을 기준으로 비교한 효과.
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
    """산술평균(제너레이터 허용, 빈 입력이면 ZeroDivisionError)."""
    xs = list(xs)
    return sum(xs) / len(xs)


def hand_table():
    """8명 표에서 ATE·ATT·ATC·단순 비교·선택 편향을 손계산과 같은 방식으로 구한다."""
    # 개인 효과 Y1 - Y0는 현실에서 관찰 불가능하고, 이 표에서만 보인다
    ate = mean(y1 - y0 for _, y0, y1, _ in CUSTOMERS)
    att = mean(y1 - y0 for _, y0, y1, t in CUSTOMERS if t == 1)
    atc = mean(y1 - y0 for _, y0, y1, t in CUSTOMERS if t == 0)
    # 우리가 실제로 보는 값: 받은 사람은 Y1, 안 받은 사람은 Y0
    treated_obs = mean(y1 for _, y0, y1, t in CUSTOMERS if t == 1)
    control_obs = mean(y0 for _, y0, y1, t in CUSTOMERS if t == 0)
    naive = treated_obs - control_obs
    # 분해: naive = ATT + (처치군의 Y0 평균 - 대조군의 Y0 평균). 뒤 항이 선택 편향(받은 사람이 원래 더 쓰는 사람)
    bias = mean(y0 for _, y0, y1, t in CUSTOMERS if t == 1) - mean(y0 for _, y0, y1, t in CUSTOMERS if t == 0)
    return {
        "ate": ate, "att": att, "atc": atc,
        "treated_obs": treated_obs, "control_obs": control_obs,
        "naive": naive, "selection_bias": bias,
    }


# ---------- 시뮬레이션: 열성도 h가 높은 사람이 쿠폰을 더 받는다 ----------
def normal(rng, sd=1.0):
    """Box-Muller 표준정규 x sd. 1 - random()으로 log(0)을 피한다."""
    u1 = 1.0 - rng.random()
    u2 = rng.random()
    return sd * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def make_population(n=20000, seed=3):
    """각 사람의 (h, Y0, Y1). 효과 tau = 4 - 3h: 열성도가 낮은 사람에게 더 크다. 평균 효과는 2.5."""
    rng = random.Random(seed)
    # 효과가 사람마다 다른(이질적) 모형이라 ATE와 ATT가 달라진다: h가 클수록 효과가 작다(모형의 선택)
    pop = []
    for _ in range(n):
        h = rng.random()
        y0 = 4.0 + 16.0 * h + normal(rng, 2.0)
        y1 = y0 + 4.0 - 3.0 * h  # E[h] = 0.5라서 평균 효과 4 - 1.5 = 2.5
        pop.append((h, y0, y1))
    return pop


def self_selected(pop, seed=4):
    """쿠폰을 직접 받아 가는 구조: 받을 확률이 0.1+0.8h."""
    rng = random.Random(seed)
    # 열성도 h가 높을수록 받을 확률이 높다. h는 Y0도 올리므로(16h) 선택 편향이 양수로 생긴다.
    return [1 if rng.random() < 0.1 + 0.8 * h else 0 for h, _, _ in pop]


def randomized(pop, seed=5):
    """h와 무관한 동전 배정(확률 0.5). 같은 pop에 대해 자기 선택과 나란히 비교하는 대조다."""
    rng = random.Random(seed)
    return [1 if rng.random() < 0.5 else 0 for _ in pop]


def estimates(pop, t):
    """배정 벡터 t(0/1)에 대한 ATE, ATT, 단순 비교(naive), 선택 편향, 처치 비율을 계산한다."""
    # ATE와 ATT는 Y0, Y1을 둘 다 쓰는 "신의 시점" 값이고, naive만 현실에서 계산 가능하다
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
    z = [1 if rng.random() < 0.5 else 0 for _ in pop]  # z: 배정
    # 배정군만 쿠폰을 쓸 수 있고(대조군 used=0), 사용 여부는 h와 무관하게 확률 takeup으로 정한다.
    # 이 무관 가정이 있어야 ITT / takeup이 전체 평균 효과(ATE)와 같아진다.
    used = [1 if (zi == 1 and rng.random() < takeup) else 0 for zi in z]
    y = [(y1 if u else y0) for (_, y0, y1), u in zip(pop, used)]
    itt = mean(yi for yi, zi in zip(y, z) if zi == 1) - mean(yi for yi, zi in zip(y, z) if zi == 0)
    # 배정군의 사용 비율로 나눠 희석(0.6배)을 되돌린다
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
