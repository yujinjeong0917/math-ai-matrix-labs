"""모델 평가 8장의 합성 데이터. 모두 고정 시드다.

이 생성 과정들은 지표가 엇갈리도록 고른 것이다. 현실에서 이런 일이 얼마나 자주 생기는지는 말해 주지 않는다.
"""

import numpy as np
from math import sqrt


def first_screen():
    """첫 화면 손 계산 예. 검사 100건 중 실제 양성(사기) 1건.
    게으른 모델: 모두 음성.  쓸 만한 모델: 그 1건을 잡고, 정상 4건을 잘못 의심한다."""
    y = np.zeros(100, dtype=int); y[0] = 1
    lazy = np.zeros(100, dtype=int)
    useful = np.zeros(100, dtype=int); useful[[0, 1, 2, 3, 4]] = 1
    return y, lazy, useful


def chicco_use_case_a1():
    """Chicco·Jurman(2020) Use case A1: 양성 91, 음성 9. TP=90, FN=1, TN=0, FP=9 -> [[TP, FN], [FP, TN]]."""
    return np.array([[90, 1], [9, 0]])


def saito_er_minus():
    """Saito·Rehmsmeier(2015) 그림 5A의 ER- 두 점: TP 500(양성 1000개 중), FP 160(음성 1000개) 대 FP 1600(음성 10000개)."""
    return np.array([[500, 500], [160, 840]]), np.array([[500, 500], [1600, 8400]])


def sklearn_ap_doc_example():
    """scikit-learn average_precision_score 문서의 예. 문서의 출력 0.83(= 5/6)."""
    return np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])


def _labels(n_pos, n_neg, rng):
    y = np.r_[np.ones(n_pos, dtype=int), np.zeros(n_neg, dtype=int)]
    return y[rng.permutation(len(y))]


def case_imbalanced_constant(p_pos=0.01, n=10_000, seed=0):
    """(a) 양성 비율 p_pos(정확히 round(p_pos n)개), '항상 음성' 분류기."""
    rng = np.random.default_rng(seed)
    n_pos = int(round(p_pos * n))
    y = _labels(n_pos, n - n_pos, rng)
    return y, np.zeros(n, dtype=int)


def case_majority_positive(p_pos=0.9, n=1000, seed=0):
    """(b) 양성이 다수(90%), '항상 양성' 분류기."""
    rng = np.random.default_rng(seed)
    n_pos = int(round(p_pos * n))
    y = _labels(n_pos, n - n_pos, rng)
    return y, np.ones(n, dtype=int)


def scores_binormal(y, rng, mu_pos=1.0, sd_pos=1.0):
    """음성 점수 N(0, 1), 양성 점수 N(mu_pos, sd_pos^2)."""
    s = rng.standard_normal(len(y))
    pos = y == 1
    s[pos] = mu_pos + sd_pos * s[pos]
    return s


def case_prevalence(prev, n_pos=1000, seed=0):
    """(c) 같은 점수 분포(양성 N(1,1), 음성 N(0,1))에서 양성 수는 고정하고 음성 수로 양성 비율을 맞춘다."""
    rng = np.random.default_rng(seed)
    n_neg = int(round(n_pos * (1 - prev) / prev))
    y = _labels(n_pos, n_neg, rng)
    return y, scores_binormal(y, rng)


# 측정 1에 쓰는 두 분류기. 이론 ROC-AUC = Phi(mu / sqrt(1 + sd^2))가 같도록 맞췄다.
#   A: 양성 N(1, 1)          -> AUC = Phi(1/sqrt(2))       ~ 0.760
#   B: 양성 N(sqrt(5/2), 2^2) -> AUC = Phi(sqrt(5/2)/sqrt(5)) = Phi(1/sqrt(2))
# B는 양성 점수가 넓게 퍼져서 맨 위 점수 구간에 양성이 더 많이 몰린다.
PAIR = {"A": (1.0, 1.0), "B": (sqrt(5.0 / 2.0), 2.0)}


def case_pair(prev, n_pos=200, seed=0):
    """같은 사람들(같은 y)에 두 분류기의 점수를 따로 뽑는다."""
    rng = np.random.default_rng(seed)
    n_neg = int(round(n_pos * (1 - prev) / prev))
    y = _labels(n_pos, n_neg, rng)
    sa = scores_binormal(y, rng, *PAIR["A"])
    sb = scores_binormal(y, rng, *PAIR["B"])
    return y, sa, sb
