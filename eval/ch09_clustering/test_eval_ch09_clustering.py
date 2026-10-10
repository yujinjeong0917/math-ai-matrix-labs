"""모델 평가 9장 검증. `uv run pytest -q eval/ch09_clustering` 로 실행한다."""

from fractions import Fraction
from itertools import permutations

import numpy as np
import pytest

import eval_ch09_cases as cases
import eval_ch09_metrics as m
import eval_ch09_torch as tm


def test_first_screen_hand_numbers():
    t, p, s = cases.first_screen()
    assert m.raw_accuracy(t, p) == 0.0 and m.matched_accuracy(t, p) == 1.0
    n11, n10, n01, n00 = m.pair_counts(t, s)
    assert (n11 + n10 + n01 + n00, n11 + n10, n00) == (15, 6, 9)  # 쌍 15개, 정답에서 같이 6, 따로 9
    assert m.rand_index(t, s) == pytest.approx(0.6)
    assert m.expected_rand_index(t, s) == pytest.approx(0.6)  # 모두 따로 두면 어떻게 섞어도 0.6
    assert m.ari(t, s) == pytest.approx(0.0, abs=1e-12)
    assert m.ari(t, p) == 1.0
    assert m.matched_accuracy(t, s) == pytest.approx(1 / 3)
    assert m.greedy_majority_accuracy(t, s) == 1.0  # 흔한 실수: 잘게 쪼개면 100%


def test_card_fixtures():
    u, v = cases.fixture_ari_half()
    assert Fraction(m.ari(u, v)).limit_denominator(100) == Fraction(-1, 2)
    assert m.rand_index(u, v) == pytest.approx(1 / 3)
    nu, nv = cases.fixture_nmi()
    assert m.mutual_info(nu, nv) / np.log(2) == pytest.approx(1.0)  # 1비트
    assert m.nmi(nu, nv, "max") == pytest.approx(0.5)
    assert m.nmi(nu, nv, "arithmetic") == pytest.approx(2 / 3)
    X = cases.fixture_silhouette()
    assert m.silhouette_samples(X, [0, 0, 1, 1])[0] == pytest.approx(9 / 11)
    assert m.silhouette_samples(X, [0, 1, 1, 1])[1] == pytest.approx(-7 / 9)
    assert m.silhouette_samples(X, [0, 1, 1, 1])[0] == 0.0  # 한 점짜리 군집은 0


def test_vinh_expected_mutual_info():
    a, b = cases.vinh_emi_sizes()
    assert round(m.expected_mutual_info_sizes(a, b), 4) == 0.4618  # Vinh 4.2절
    # 점을 10배로 늘리면 Vinh가 적은 두 상한(0.0764, 0.0780)보다 작아야 한다
    assert m.expected_mutual_info_sizes([x * 10 for x in a], [x * 10 for x in b]) < 0.0764


def test_permutation_model_bruteforce():
    """순열 모형의 기댓값을 720가지 배치를 다 세어 확인: E{I}·E[RI] 공식, ARI·AMI 평균 0."""
    u, v = np.array([0, 0, 0, 1, 1, 2]), np.array([0, 0, 1, 1, 2, 2])
    perms = [v[list(p)] for p in permutations(range(6))]
    assert np.mean([m.mutual_info(u, w) for w in perms]) == pytest.approx(m.expected_mutual_info(u, v), abs=1e-12)
    assert np.mean([m.rand_index(u, w) for w in perms]) == pytest.approx(m.expected_rand_index(u, v), abs=1e-12)
    assert np.mean([m.ari(u, w) for w in perms]) == pytest.approx(0.0, abs=1e-12)
    assert np.mean([m.ami(u, w) for w in perms]) == pytest.approx(0.0, abs=1e-12)


def test_hungarian_matches_bruteforce():
    rng = np.random.default_rng(1)
    for _ in range(100):
        w = rng.integers(0, 30, (rng.integers(1, 8), rng.integers(1, 8)))
        r, c = m.hungarian_max(w)
        assert len(set(r)) == len(r) and len(set(c)) == len(c)  # 일대일
        assert w[r, c].sum() == m.brute_force_max(w)


def test_relabel_invariance():
    X, y, c = cases.case_permuted_blobs(seed=0)
    assert m.raw_accuracy(y, c) == 0.0
    rng = np.random.default_rng(3)
    noisy = c.copy()
    flip = rng.random(len(c)) < 0.2
    noisy[flip] = rng.integers(0, 3, int(flip.sum()))
    base = (m.ari(y, noisy), m.nmi(y, noisy), m.ami(y, noisy), m.matched_accuracy(y, noisy))
    for perm in permutations(range(3)):
        relab = np.array(perm)[noisy] + 10  # 번호 값도 아무렇게나
        now = (m.ari(y, relab), m.nmi(y, relab), m.ami(y, relab), m.matched_accuracy(y, relab))
        assert np.allclose(now, base, atol=1e-12)
    for f in (m.ari, m.nmi, m.ami, m.matched_accuracy):
        assert f(y, c) == pytest.approx(1.0)


def test_symmetry_and_ranges():
    rng = np.random.default_rng(4)
    for _ in range(30):
        u, v = rng.integers(0, 4, 60), rng.integers(0, 6, 60)
        assert m.ari(u, v) == pytest.approx(m.ari(v, u))
        assert m.ami(u, v) == pytest.approx(m.ami(v, u))
        assert 0.0 <= m.nmi(u, v) <= 1.0 and m.ari(u, v) <= 1.0 and m.ami(u, v) <= 1.0


def test_ari_pair_formula_matches_vinh_form():
    """Vinh 2절에 적힌 쌍 세기 꼴 2(N00 N11 - N01 N10) / (...)와 같은 값인지."""
    rng = np.random.default_rng(5)
    for _ in range(30):
        u, v = rng.integers(0, 3, 40), rng.integers(0, 5, 40)
        n11, n10, n01, n00 = m.pair_counts(u, v)
        alt = 2 * (n00 * n11 - n01 * n10) / ((n00 + n01) * (n01 + n11) + (n00 + n10) * (n10 + n11))
        assert m.ari(u, v) == pytest.approx(alt, abs=1e-12)


def test_random_labels_chance_correction():
    """n=100에서 무작위 라벨: 군집 수가 늘면 NMI는 오르고, ARI·AMI 평균은 0 근처."""
    means = {}
    for k in (2, 10, 50):
        vals = np.array([[m.nmi(t, p), m.ari(t, p), m.ami(t, p)]
                         for t, p in (cases.case_random_labels(n=100, k=k, seed=s) for s in range(30))])
        means[k] = vals.mean(axis=0)
    assert means[2][0] < means[10][0] < means[50][0]
    assert means[50][0] > 0.2
    for k in means:
        assert abs(means[k][1]) < 0.02 and abs(means[k][2]) < 0.02


def test_silhouette_prefers_wrong_split_on_moons():
    X, y = cases.case_moons(seed=0)
    km = m.kmeans(X, 2, seed=0)
    assert m.silhouette(X, km) > m.silhouette(X, y)
    assert m.ari(y, km) < 0.5


def test_silhouette_range_and_validity():
    X, y = cases.case_blobs(k=4, seed=0)
    s = m.silhouette_samples(X, y)
    assert np.all(s >= -1) and np.all(s <= 1)
    with pytest.raises(ValueError):
        m.silhouette(X, np.zeros(len(X), dtype=int))


def test_kmeans_recovers_blobs():
    X, y, _ = cases.case_permuted_blobs(seed=0)
    assert m.matched_accuracy(y, m.kmeans(X, 3, seed=0)) == 1.0


def test_numpy_matches_torch():
    rng = np.random.default_rng(6)
    for _ in range(10):
        n = int(rng.integers(30, 150))
        u, v = rng.integers(0, 4, n), rng.integers(0, 7, n)
        P = rng.standard_normal((n, 3))
        assert abs(m.ari(u, v) - tm.ari(u, v)) < 1e-12
        assert abs(m.mutual_info(u, v) - tm.mutual_info(u, v)) < 1e-12
        assert abs(m.expected_mutual_info(u, v) - tm.expected_mutual_info(u, v)) < 1e-12
        assert abs(m.ami(u, v, "arithmetic") - tm.ami(u, v, "arithmetic")) < 1e-10
        assert abs(m.silhouette(P, u) - tm.silhouette(P, u)) < 1e-12


def test_web_chapter_exercise_answers():
    # 연습 1: 정답 (가 가 나 나 다 다), 군집 (1 1 1 2 2 3): 번호 맞춘 정확도 4/6
    t = np.array([0, 0, 1, 1, 2, 2]); c = np.array([0, 0, 0, 1, 1, 2])
    assert m.matched_accuracy(t, c) == pytest.approx(4 / 6)
    # 연습 2: 정답 (가 가 나 나), 모두 따로 -> RI 4/6, 우연 평균 4/6, ARI 0
    t2, s2 = np.array([0, 0, 1, 1]), np.arange(4)
    assert m.rand_index(t2, s2) == pytest.approx(4 / 6) and m.ari(t2, s2) == pytest.approx(0.0, abs=1e-12)
    # 연습 3: 수직선 0, 1, 5에서 {0, 1}, {5}: 점 1의 실루엣 (4 - 1) / 4 = 0.75
    X = np.array([[0.0], [1.0], [5.0]])
    assert m.silhouette_samples(X, [0, 0, 1])[1] == pytest.approx(0.75)
