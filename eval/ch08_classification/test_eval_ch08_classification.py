"""모델 평가 8장 검증. `uv run pytest -q eval/ch08_classification` 로 실행한다."""

import math
from fractions import Fraction

import numpy as np
import pytest

import eval_ch08_cases as cases
import eval_ch08_metrics as m
import eval_ch08_torch as tm


def test_first_screen_hand_numbers():
    y, lazy, useful = cases.first_screen()
    cl, cu = m.confusion(y, lazy), m.confusion(y, useful)
    assert cl.tolist() == [[0, 1], [0, 99]] and cu.tolist() == [[1, 0], [4, 95]]
    assert m.accuracy(cl) == 0.99 and m.accuracy(cu) == 0.96  # 정확도는 게으른 모델 편
    assert m.recall(cl) == 0.0 and math.isnan(m.precision(cl))  # 양성이라고 말한 적이 없어 정밀도는 0/0
    assert m.f1(cl) == 0.0
    assert Fraction(m.f1(cu)).limit_denominator(100) == Fraction(1, 3)  # 2*1 / (2*1 + 4 + 0)
    assert m.mcc(cu) == pytest.approx(95 / math.sqrt(5 * 1 * 99 * 95))
    assert round(m.mcc(cu), 3) == 0.438
    assert math.isnan(m.mcc(cl)) and m.mcc(cl, "chicco") == 0.0


def test_chicco_use_case_a1():
    """Chicco·Jurman(2020) Use case A1: accuracy 0.90, F1 0.95, MCC -0.03."""
    a1 = cases.chicco_use_case_a1()
    assert round(m.accuracy(a1), 2) == 0.90
    assert round(m.f1(a1), 2) == 0.95
    assert round(m.mcc(a1), 2) == -0.03


def test_mcc_undefined_policies():
    # 한 열이 0 (항상 양성), 한 행이 0 (실제 음성 없음) -> chicco 0
    for cm in ([[900, 0], [100, 0]], [[0, 10], [0, 90]], [[5, 3], [0, 0]], [[0, 0], [4, 7]]):
        assert math.isnan(m.mcc(cm)) and m.mcc(cm, "chicco") == 0.0
    # 0이 아닌 칸이 하나뿐: 다 맞히면 +1, 다 틀리면 -1
    assert m.mcc([[7, 0], [0, 0]], "chicco") == 1.0 and m.mcc([[0, 0], [0, 7]], "chicco") == 1.0
    assert m.mcc([[0, 7], [0, 0]], "chicco") == -1.0 and m.mcc([[0, 0], [7, 0]], "chicco") == -1.0
    with pytest.raises(ValueError):
        m.mcc([[1, 0], [0, 0]], "zero?")


def test_mcc_range_and_extremes():
    y = np.array([1, 1, 0, 0, 0])
    assert m.mcc(m.confusion(y, y)) == 1.0
    assert m.mcc(m.confusion(y, 1 - y)) == -1.0
    rng = np.random.default_rng(0)
    for _ in range(200):
        cm = rng.integers(1, 50, (2, 2))
        assert -1.0 <= m.mcc(cm) <= 1.0


def test_f_equals_one_minus_e():
    """van Rijsbergen(1979) 7장: alpha = 1/(beta^2 + 1)이면 1 - E = F_beta."""
    for p in (0.1, 0.4, 0.9):
        for r in (0.2, 0.7, 1.0):
            for beta in (0.5, 1.0, 2.0):
                fb = (1 + beta ** 2) * p * r / (beta ** 2 * p + r)
                assert 1 - m.van_rijsbergen_e(p, r, 1 / (beta ** 2 + 1)) == pytest.approx(fb, abs=1e-12)


def test_e_beta1_is_normalised_symmetric_difference():
    """같은 장의 특수한 경우 (1): beta = 1이면 E = |A sym B| / (|A| + |B|) = (FP + FN) / (2TP + FP + FN)."""
    rng = np.random.default_rng(1)
    for _ in range(50):
        y, yhat = rng.integers(0, 2, 40), rng.integers(0, 2, 40)
        A, B = set(np.nonzero(y)[0]), set(np.nonzero(yhat)[0])  # A = 관련 문서, B = 검색된 문서
        if not A or not B or not (A & B):
            continue
        cm = m.confusion(y, yhat)
        e = m.van_rijsbergen_e(m.precision(cm), m.recall(cm), 0.5)
        assert e == pytest.approx(len(A ^ B) / (len(A) + len(B)), abs=1e-12)
        assert 1 - e == pytest.approx(m.f1(cm), abs=1e-12)


def test_f1_ignores_tn():
    """TN만 바꾸면 F1은 그대로, MCC와 정확도는 바뀐다."""
    a, b = np.array([[50, 10], [20, 5]]), np.array([[50, 10], [20, 5000]])
    assert m.f1(a) == m.f1(b)
    assert m.mcc(a) != pytest.approx(m.mcc(b)) and m.accuracy(a) != m.accuracy(b)


def test_majority_positive_case():
    y, yhat = cases.case_majority_positive()
    cm = m.confusion(y, yhat)
    assert round(m.f1(cm), 3) == 0.947 and math.isnan(m.mcc(cm)) and m.mcc(cm, "chicco") == 0.0


def test_imbalanced_constant_case():
    y, yhat = cases.case_imbalanced_constant(0.01, 10_000, seed=0)
    cm = m.confusion(y, yhat)
    assert cm.tolist() == [[0, 100], [0, 9900]]
    assert m.accuracy(cm) == 0.99 and m.recall(cm) == 0.0 and m.f1(cm) == 0.0


def test_ap_matches_sklearn_doc_example():
    """scikit-learn average_precision_score 문서의 예: 0.83 (= 5/6). 사다리꼴 넓이는 다르다."""
    y, s = cases.sklearn_ap_doc_example()
    assert m.average_precision_np(y, s) == pytest.approx(5 / 6, abs=1e-12)
    assert tm.average_precision(y, s) == pytest.approx(5 / 6, abs=1e-12)
    assert m.pr_auc_trapezoid(y, s) != pytest.approx(5 / 6)


def test_ap_and_auc_with_ties():
    """동점 점수는 한 임계값으로 묶어야 한다. 모든 점수가 같으면 AP = 양성 비율, AUC = 0.5."""
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    s = np.zeros(10)
    assert m.average_precision_np(y, s) == pytest.approx(0.2)
    assert m.roc_auc_np(y, s) == pytest.approx(0.5)
    assert tm.average_precision(y, s) == pytest.approx(0.2)


def test_roc_auc_equals_pairwise_count():
    rng = np.random.default_rng(2)
    for _ in range(30):
        n = int(rng.integers(10, 200))
        y = (rng.random(n) < 0.3).astype(int)
        if y.sum() in (0, n):
            continue
        s = np.round(rng.standard_normal(n) + y, 1)
        assert m.roc_auc_np(y, s) == pytest.approx(m.auc_pairwise(y, s), abs=1e-12)
        assert m.roc_auc_np(y, s) == pytest.approx(tm.roc_auc_pairs(y, s), abs=1e-12)


def test_replicating_negatives_keeps_roc_auc_but_lowers_ap():
    """음성을 k번 복제하면 FPR은 그대로라 ROC-AUC는 정확히 같고, FP가 k배가 되어 AP는 떨어진다."""
    y, s = cases.case_prevalence(0.5, 300, seed=4)
    neg = y == 0
    for k in (3, 20):
        y2 = np.r_[y[~neg], np.zeros(k * neg.sum(), dtype=int)]
        s2 = np.r_[s[~neg], np.tile(s[neg], k)]
        assert m.roc_auc_np(y2, s2) == pytest.approx(m.roc_auc_np(y, s), abs=1e-12)
        assert m.average_precision_np(y2, s2) < m.average_precision_np(y, s) - 0.2


def test_pair_has_equal_theoretical_auc():
    from math import erf, sqrt
    phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))  # noqa: E731
    (ma, sa), (mb, sb) = cases.PAIR["A"], cases.PAIR["B"]
    assert phi(ma / sqrt(1 + sa ** 2)) == pytest.approx(phi(mb / sqrt(1 + sb ** 2)), abs=1e-15)


def test_numpy_torch_agree():
    rng = np.random.default_rng(3)
    for _ in range(10):
        n = int(rng.integers(30, 300))
        y = (rng.random(n) < 0.3).astype(int)
        if y.sum() in (0, n):
            continue
        s = np.round(rng.standard_normal(n) + y, 1)
        yhat = (s > 0.5).astype(int)
        cm = m.confusion(y, yhat)
        assert np.array_equal(cm, tm.confusion(y, yhat).numpy().astype(int))
        assert m.f_beta(cm, 2.0) == pytest.approx(tm.f_beta(cm, 2.0), abs=1e-12)
        assert m.mcc(cm) == pytest.approx(tm.mcc(cm), abs=1e-12)
        assert m.average_precision_np(y, s) == pytest.approx(tm.average_precision(y, s), abs=1e-12)
        p = 1 / (1 + np.exp(-s))
        assert m.brier(y, p) == pytest.approx(tm.brier(y, p), abs=1e-12)


def test_brier_honest_forecast_is_best():
    """사건 확률이 0.01이면 기대 브라이어 점수 0.01(1-q)^2 + 0.99 q^2는 q = 0.01에서 가장 작다."""
    qs = np.linspace(0, 0.1, 1001)
    exp = 0.01 * (1 - qs) ** 2 + 0.99 * qs ** 2
    assert qs[np.argmin(exp)] == pytest.approx(0.01)
    assert m.brier([1, 0, 0, 0], [0.5, 0.5, 0.0, 0.0]) == pytest.approx((0.25 + 0.25) / 4)


def test_exercise_answers():
    # 연습 1: 양성 20, 음성 980. TP 15, FP 45 -> 정밀도 0.25, 재현율 0.75, F1 = 30/(30+45+5) = 0.375
    cm = np.array([[15, 5], [45, 935]])
    assert (m.precision(cm), m.recall(cm), m.f1(cm)) == (0.25, 0.75, 0.375)
    assert m.accuracy(cm) == 0.95
    # 연습 2: F1은 그대로, TN만 935 -> 9935로 늘면 MCC는 오른다
    cm2 = np.array([[15, 5], [45, 9935]])
    assert m.f1(cm2) == m.f1(cm) and m.mcc(cm2) > m.mcc(cm)
    # 연습 3: 점수 (0.9, 0.8, 0.7, 0.6), 정답 (1, 0, 1, 0) -> AP = 0.5*1 + 0.5*(2/3) = 5/6
    assert m.average_precision_np([1, 0, 1, 0], [0.9, 0.8, 0.7, 0.6]) == pytest.approx(5 / 6)
