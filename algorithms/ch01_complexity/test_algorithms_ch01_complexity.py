"""알고리즘 1장 검증. `uv run pytest -q algorithms/ch01_complexity` 로 실행한다.

설계도는 Hypothesis 성질 테스트를 적었지만 의존성을 늘리지 않으려고, 시드를 고정한 무작위 입력
수백 개를 도는 반복문으로 같은 성질을 확인한다.
"""

import math
import random

import algorithms_ch01_dup as dup
import algorithms_ch01_timing as tm

METHODS = list(dup.METHODS)


def test_first_screen_hand_numbers():
    # 첫 화면: 4명이면 4*3/2 = 6번, 8명이면 8*7/2 = 28번, 16명이면 120번
    assert [dup.count_ops("pairs", list(range(n))) for n in (4, 8, 16)] == [6, 28, 120]
    assert 28 / 6 > 4 and 120 / 28 > 4  # 두 배로 늘리면 네 배 남짓
    assert dup.count_ops("set", list(range(8))) == 8


def test_all_methods_agree_on_random_lists():
    rng = random.Random(0)
    for _ in range(500):
        n = rng.randint(0, 30)
        xs = [rng.randint(0, 2 * n + 1) for _ in range(n)]  # 값 범위를 좁혀 중복이 자주 나오게
        truth = len(set(xs)) < len(xs)
        for m in METHODS:
            assert dup.METHODS[m](xs) is truth, (m, xs)
            assert dup.METHODS[m](xs, dup.Counter()) is truth, (m, xs)


def test_pairs_count_is_exactly_n_choose_2_on_distinct_inputs():
    for n in list(range(0, 40)) + [100, 500]:
        for seed in range(3):
            assert dup.count_ops("pairs", dup.distinct_list(n, seed)) == n * (n - 1) // 2


def test_counts_are_deterministic_and_order_free_where_expected():
    xs = dup.distinct_list(300, 1)
    for m in METHODS:
        assert len({dup.count_ops(m, xs) for _ in range(5)}) == 1  # 같은 입력이면 매번 같다
    rng = random.Random(2)
    for _ in range(20):
        ys = xs[:]
        rng.shuffle(ys)
        assert dup.count_ops("pairs", ys) == dup.count_ops("pairs", xs)  # 순서를 섞어도 같다
        assert dup.count_ops("set", ys) == dup.count_ops("set", xs)
    # 정렬 기반은 입력 순서에 따라 달라진다(이미 정렬된 입력은 비교가 훨씬 적다)
    assert dup.count_ops("sorted", sorted(xs)) < dup.count_ops("sorted", xs)


def test_msort_comparisons_within_n_log_n():
    for n in (2, 3, 7, 64, 1000):
        c = dup.Counter()
        out = dup._merge_sort(dup.distinct_list(n, 0), c)
        assert out == sorted(out)
        assert c.comparisons <= n * math.ceil(math.log2(n))


def test_counter_does_not_change_answer_with_early_duplicate():
    xs = [5, 5] + list(range(100, 200))
    c = dup.Counter()
    assert dup.has_dup_pairs(xs, c) and c.comparisons == 1  # 첫 쌍에서 바로 끝난다


def test_timing_summary_shape():
    ts, number = tm.time_call(lambda: sum(range(10)), repeats=4, target=0.001)
    assert len(ts) == 4 and number >= 1 and all(t > 0 for t in ts)
    s = tm.summarize([1.0, 2.0, 3.0, 4.0])
    assert s["min"] == 1.0 and s["max"] == 4.0 and s["median"] == 2.5 and s["max_over_min"] == 4.0


def test_page_derived_numbers():
    # 웹 챕터에 쓴 결정적 파생 숫자들 (시간에서 나온 비율은 실행마다 바뀌므로 여기서 검사하지 않는다)
    assert 499500 / 1000 == 499.5  # "횟수는 500배 가까이"
    assert 4950 / 45 == 110
    assert 49995000 / 499500 > 100 and round(49995000 / 499500) == 100
    assert 120 / (16 * math.log2(16)) == 1.875
    assert round((16384 * 16383 / 2) / (16384 * 14)) == 585
    ms = dup.distinct_list(64, 0)
    assert dup.count_ops("msort", ms) == 370 and 370 / 2016 < 0.2  # "5분의 1도 안"
    for n in range(2, 2000):
        assert n * n / 4 <= n * (n - 1) / 2 <= n * n / 2  # c=1/4, C=1/2, n0=2
    assert 16 / 4 <= 6 <= 16 / 2
