"""모델 평가 7장의 판독 사례. 모두 고정 시드의 합성 데이터다.

사례마다 make_case_<이름>(seed)가 dict를 돌려준다.
  y_train, y_test : 학습 구간(MASE 분모용)과 평가 구간의 정답
  A, B            : 두 후보의 평가 구간 예측
  objective       : 목적 손실 이름. objective_loss(case, 'A'|'B')로 계산한다(작을수록 좋다).
  alt_objective   : (있으면) 목적을 바꾸면 답이 바뀌는지 보여 주는 두 번째 목적
  intended_flag   : 이 사례가 일부러 만든 엇갈림. 판독 절차(eval_ch07_casebook.read_table)가 켜야 하는 깃발
이 생성 과정은 엇갈림이 나오도록 설계한 것이다. 현실에서 이런 엇갈림이 얼마나 자주 나오는지는 말해 주지 않는다.
"""

import numpy as np

CASE_NAMES = ("control", "spiky", "small_values", "flat_level", "monthly_bias", "fixable_bias", "extrapolation")


def first_screen():
    """첫 화면 손 계산 예. 배달 5건의 도착 시간 예측 오차(분).
    A는 네 번 1분, 한 번 11분 틀리고, B는 다섯 번 모두 4분 틀린다."""
    return np.array([1.0, 1.0, 1.0, 1.0, 11.0]), np.array([4.0, 4.0, 4.0, 4.0, 4.0])


def fixture_small():
    """손 계산 fixture. 학습 4개, 평가 4개.
    y_train = 10, 12, 11, 13 -> naive 오차 2, 1, 2 -> 분모 5/3
    y_test  = 10, 20, 10, 20 (평균 15, SST = 100)
    A = 11, 19, 11, 19 -> e = -1, 1, -1, 1: MAE 1, RMSE 1, R^2 = 1 - 4/100 = 0.96, MAPE = 100*(0.1+0.05+0.1+0.05)/4 = 7.5,
        MASE = 1/(5/3) = 0.6, NMBE = 0
    B = 8, 18, 8, 18 -> e = 2, 2, 2, 2: MAE 2, RMSE 2, R^2 = 1 - 16/100 = 0.84, MAPE = 100*(0.2+0.1+0.2+0.1)/4 = 15,
        MASE = 2/(5/3) = 1.2, NMBE = 100*8/(4*15) = 13.33..."""
    return (np.array([10.0, 12.0, 11.0, 13.0]), np.array([10.0, 20.0, 10.0, 20.0]),
            np.array([11.0, 19.0, 11.0, 19.0]), np.array([8.0, 18.0, 8.0, 18.0]))


def _hourly_load(rng, n, level=100.0, amp=10.0, sd=3.0):
    """시간 단위 부하 y_t = level + amp sin(2 pi t / 24) + N(0, sd^2)."""
    t = np.arange(n)
    return level + amp * np.sin(2 * np.pi * t / 24.0) + rng.normal(0.0, sd, n)


# ---------- 목적 손실 ----------

def big_miss_cost(e, tau=8.0, lam=5.0):
    """큰 실수가 비싼 비용: |e| + lam * max(|e| - tau, 0)의 평균. tau를 넘는 부분은 (1 + lam)배로 센다."""
    a = np.abs(e)
    return float(np.mean(a + lam * np.maximum(a - tau, 0.0)))


def objective_loss(case, which, name=None):
    """case의 목적 손실(작을수록 좋다). name을 주면 그 목적으로 계산한다."""
    name = name or case["objective"]
    y = case["y_test"]
    p = case[which]
    e = y - p
    if name == "abs_units":  # 틀린 개수(단위) 그대로의 합. 평균으로 쓴다.
        return float(np.mean(np.abs(e)))
    if name == "big_miss":
        return big_miss_cost(e)
    if name == "relative":  # 가게마다 상대 오차가 똑같이 중요할 때
        return float(np.mean(np.abs(e) / np.abs(y)))
    if name == "monthly_total":  # 달마다 합계를 얼마나 틀렸나(요금 고지서 기준)
        k = case["month_len"]
        ys = y.reshape(-1, k).sum(axis=1)
        ps = p.reshape(-1, k).sum(axis=1)
        return float(np.mean(np.abs(ys - ps)))
    if name == "abs_after_offset":  # 검증 구간의 평균 오차만큼 더해서 쓴 뒤의 평가 구간 MAE
        shift = float(np.mean(case["y_val"] - case[which + "_val"]))
        return float(np.mean(np.abs(y - (p + shift))))
    raise ValueError(f"알 수 없는 목적: {name}")


# ---------- 사례 ----------

def make_case_control(seed, n_train=200, n_test=200):
    """대조군. A와 B가 같은 과정(잡음 표준편차 3)에서 나온다. 고를 이유가 없는 상황."""
    rng = np.random.default_rng(70_000 + seed)
    y = _hourly_load(rng, n_train + n_test)
    yt = y[n_train:]
    return {"name": "control", "y_train": y[:n_train], "y_test": yt,
            "A": yt + rng.normal(0, 3.0, n_test), "B": yt + rng.normal(0, 3.0, n_test),
            "objective": "abs_units", "intended_flag": "indistinct"}


def make_case_spiky(seed, n_train=200, n_test=200, p_big=0.05, big=20.0):
    """사례 1 (1·2장). A: 대부분 표준편차 1로 작게, 5%는 크기 20으로 크게 틀림. B: 늘 표준편차 3으로 틀림.
    MAE는 A가, RMSE는 B가 낮다. 목적: 8을 넘는 실수가 비싸다(big_miss). 두 번째 목적: 틀린 만큼만 비용."""
    rng = np.random.default_rng(71_000 + seed)
    y = _hourly_load(rng, n_train + n_test)
    yt = y[n_train:]
    e_a = rng.normal(0.0, 1.0, n_test)
    k = int(round(p_big * n_test))
    idx = rng.choice(n_test, size=k, replace=False)
    e_a[idx] = big * rng.choice([-1.0, 1.0], size=k)
    return {"name": "spiky", "y_train": y[:n_train], "y_test": yt, "A": yt - e_a, "B": yt + rng.normal(0, 3.0, n_test),
            "objective": "big_miss", "alt_objective": "abs_units", "intended_flag": "spiky"}


def make_case_small_values(seed, n_train=200, n_test=200, sd_a=1.0, rel_b=0.2):
    """사례 2 (3·4장). 가게 200곳의 하루 판매량(로그정규, 로그의 평균 1.5, 표준편차 1). 작은 가게가 많고 큰 가게가 조금.
    A: 가게 크기와 상관없이 표준편차 1개만큼 틀림. B: 가게 크기의 20%만큼 틀림.
    R^2(=RMSE 순위)는 A, MAPE는 B를 고른다. 목적: 전체 몇 개를 틀렸나(abs_units). 두 번째 목적: 가게마다 상대 오차(relative)."""
    rng = np.random.default_rng(72_000 + seed)
    y = np.exp(rng.normal(1.5, 1.0, n_train + n_test))
    yt = y[n_train:]
    a = np.maximum(yt + rng.normal(0.0, sd_a, n_test), 0.0)
    b = yt * (1.0 + rng.normal(0.0, rel_b, n_test))
    return {"name": "small_values", "y_train": y[:n_train], "y_test": yt, "A": a, "B": b,
            "objective": "abs_units", "alt_objective": "relative", "intended_flag": "near_zero"}


def make_case_flat_level(seed, n_train=200, n_test=100, level=1000.0, sd=5.0, window=10):
    """사례 3 (4·5장). 수준 약 1000, 하루 변화 표준편차 5인 랜덤워크(5장 case_random_walk_level과 같은 모양).
    A: 최근 10일 평균으로 1스텝 예측(그럴듯한 '평활' 모델). B: 1스텝 naive(어제 값 그대로).
    A의 MAPE는 1% 아래인데 MASE는 1보다 크다. 목적: 1스텝 오차 개수(abs_units)."""
    rng = np.random.default_rng(73_000 + seed)
    y = level + np.cumsum(rng.normal(0.0, sd, n_train + n_test))
    yt = y[n_train:]
    a = np.array([y[t - window:t].mean() for t in range(n_train, n_train + n_test)])
    b = y[n_train - 1:-1].copy()
    return {"name": "flat_level", "y_train": y[:n_train], "y_test": yt, "A": a, "B": b,
            "objective": "abs_units", "intended_flag": "below_naive"}


def make_case_monthly_bias(seed, n_train=672, months=12, bias=0.08, sd_a=15.0, sd_b=5.0):
    """사례 4 (6장). 1년(30일 x 12달 = 8640시간) 부하, 평균 100, 하루 주기 +-30(6장 base_load와 같은 모양, 달 길이만 30일로 고정).
    A: 편향 없음, 잡음 표준편차 15(6장의 25보다 작게 해서 R^2가 0 아래로 내려가지 않게 했다). B: 늘 8% 모자람, 잡음 표준편차 5.
    RMSE는 B, |NMBE|는 A를 고른다. 목적: 달마다 합계 오차(monthly_total). 두 번째 목적: 시간 단위 오차(abs_units)."""
    rng = np.random.default_rng(74_000 + seed)
    month_len = 30 * 24
    n = n_train + months * month_len
    t = np.arange(n)
    y = 100.0 + 30.0 * np.sin(2 * np.pi * t / 24.0) + rng.normal(0.0, 5.0, n)
    yt = y[n_train:]
    return {"name": "monthly_bias", "y_train": y[:n_train], "y_test": yt,
            "A": yt + rng.normal(0.0, sd_a, len(yt)), "B": (1.0 - bias) * yt + rng.normal(0.0, sd_b, len(yt)),
            "objective": "monthly_total", "alt_objective": "abs_units", "month_len": month_len, "intended_flag": "bias"}


def make_case_fixable_bias(seed, n_train=672, n_val=720, n_test=720, offset=9.0, sd_a=3.0, sd_b=8.0):
    """사례 5 (6장). 부하는 평균 100, 하루 주기 +-30. 한 달(720시간) 평가. A: 늘 9만큼 모자람(수준 100의 9%), 잡음 표준편차 3. B: 편향 없음, 잡음 표준편차 8.
    보정 전 지표는 모두 B를 고른다. 목적: 바로 앞 검증 달의 평균 오차만큼 더해서 쓴 뒤의 오차(abs_after_offset)."""
    rng = np.random.default_rng(75_000 + seed)
    y = _hourly_load(rng, n_train + n_val + n_test, amp=30.0)
    yv, yt = y[n_train:n_train + n_val], y[n_train + n_val:]
    return {"name": "fixable_bias", "y_train": y[:n_train], "y_test": yt, "y_val": yv,
            "A": yt - offset + rng.normal(0, sd_a, n_test), "B": yt + rng.normal(0, sd_b, n_test),
            "A_val": yv - offset + rng.normal(0, sd_a, n_val), "B_val": yv + rng.normal(0, sd_b, n_val),
            "objective": "abs_after_offset", "intended_flag": "bias"}


def make_case_extrapolation(seed, n_train=50, n_test=50, noise=1.0):
    """사례 6 (3장). x를 [0, 5]에서 뽑아 y = x^2 + 잡음에 맞춘 뒤, [5, 10]에서 잰다(3장 case_extrapolation과 같은 설정).
    A: 직선. B: 2차식. 평가 R^2는 A가 0보다 작다. 목적: 틀린 크기(abs_units).
    MASE 분모를 위해 학습·평가 모두 x 순서로 정렬해 '계열'처럼 다룬다."""
    rng = np.random.default_rng(76_000 + seed)
    xtr = np.sort(rng.uniform(0.0, 5.0, n_train))
    ytr = xtr ** 2 + rng.normal(0.0, noise, n_train)
    xte = np.sort(rng.uniform(5.0, 10.0, n_test))
    yte = xte ** 2 + rng.normal(0.0, noise, n_test)
    c1 = np.polyfit(xtr, ytr, 1)
    c2 = np.polyfit(xtr, ytr, 2)
    return {"name": "extrapolation", "y_train": ytr, "y_test": yte, "A": np.polyval(c1, xte), "B": np.polyval(c2, xte),
            "A_train_fit": np.polyval(c1, xtr), "objective": "abs_units", "intended_flag": "below_mean"}


MAKERS = {"control": make_case_control, "spiky": make_case_spiky, "small_values": make_case_small_values,
          "flat_level": make_case_flat_level, "monthly_bias": make_case_monthly_bias,
          "fixable_bias": make_case_fixable_bias, "extrapolation": make_case_extrapolation}
