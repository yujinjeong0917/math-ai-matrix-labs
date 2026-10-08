"""알고리즘 2장 검증. `uv run pytest -q algorithms/ch02_binary_search` 로 실행한다.

설계도는 Hypothesis 성질 테스트를 적었지만 의존성을 늘리지 않으려고, 작은 입력 전수 검사와
시드를 고정한 무작위 입력 반복문으로 같은 성질을 확인한다.
"""

import bisect
import math
import random
import sys

import numpy as np
import torch

import algorithms_ch02_bsearch as bs

torch.set_num_threads(2)
CARDS = [3, 5, 8, 12, 15, 19, 23, 30]


def test_first_screen_hand_walk():
    # 카드 8장에서 16 이상인 첫 카드: [0,8) -> mid 4(15<16) -> [5,8) -> mid 6(23) -> [5,6) -> mid 5(19) -> [5,5)
    r = bs.run_variant("d_correct", CARDS, 16, trace=True)
    states = [(s["state"]["lo"], s["state"]["hi"], s["state"]["mid"]) for s in r["trace"]]
    assert states == [(0, 8, None), (5, 8, 4), (5, 6, 6), (5, 5, 5)]
    assert r["answer"] == 5 and CARDS[5] == 19 and r["steps"] == 3
    lin = bs.Counter()
    assert bs.linear_lower_bound(CARDS, 16, lin) == 5 and lin.comparisons == 6  # 앞에서부터면 여섯 번
    # 0~5장, 값 5종(중복 허용) 정렬 더미 252가지 x 목표값 11가지 = 2,772가지
    assert sum(math.comb(n + 4, 4) for n in range(6)) == len(list(bs.sorted_arrays(5))) == 252
    assert 252 * len(bs.targets()) == 2772
    # lo = mid 판은 후보가 15 한 장 남은 [4,5) 에서 멈추지 않는다
    s = bs.run_variant("b_lo_mid", CARDS, 16, trace=True)
    assert s["outcome"] == "no_stop"
    tail = [(t["state"]["lo"], t["state"]["hi"]) for t in s["trace"][3:7]]
    assert tail == [(4, 5)] * 4
    # 질문 수 상한: 8장이면 4번, 100만 장이면 20번
    assert bs.max_steps_bound(8) == 4 and bs.max_steps_bound(10 ** 6) == 20
    assert 2 ** 19 < 10 ** 6 + 1 <= 2 ** 20


def test_overflow_numbers():
    d = bs.mid_overflow_demo()
    assert d["true_sum"] == 3_500_000_000 > d["int32_max"] == 2_147_483_647
    assert d["wrapped_sum"] == 3_500_000_000 - 2 ** 32 == -794_967_296
    assert d["naive_mid"] == -397_483_648 < 0
    assert d["safe_mid"] == d["python_int_mid"] == 1_750_000_000
    assert d["array_wrapped_sum"] == d["wrapped_sum"]


def test_lower_bound_matches_bisect_and_numpy_and_torch():
    rng = random.Random(0)
    for _ in range(2000):
        n = rng.randint(0, 40)
        a = sorted(rng.randint(0, 20) for _ in range(n))
        x = rng.randint(-1, 21)
        c = bs.Counter()
        got = bs.lower_bound(a, x, c)
        assert got == bisect.bisect_left(a, x) == bs.linear_lower_bound(a, x)
        assert c.steps <= bs.max_steps_bound(n)
        if n:
            assert got == int(np.searchsorted(np.array(a), x, side="left"))
            assert got == int(torch.searchsorted(torch.tensor(a), torch.tensor(x), right=False))


def test_invariant_holds_and_size_shrinks_every_step():
    rng = random.Random(1)
    for _ in range(500):
        n = rng.randint(0, 30)
        a = sorted(rng.randint(0, 10) for _ in range(n))
        x = rng.randint(-1, 11)
        t = bs.run_variant("d_correct", a, x, trace=True)
        for row in t["trace"]:
            assert all(i["holds"] for i in row["invariants"]), (a, x, row)
            assert row["state"].get("shrank", True)


def test_exhaustive_variants_and_shortest_counterexamples():
    r = bs.exhaustive(5)
    assert r["d_correct"]["fail"] == 0
    assert r["a_le_with_hi_mid"]["counts"]["ok"] == 0  # 한 번도 정상 종료하지 못한다
    assert r["a_le_with_hi_mid"]["first_failure"]["a"] == []  # 빈 배열에서 a[0] 을 읽는다
    assert r["b_lo_mid"]["first_failure"] == {"a": [0], "x": 1, "outcome": "no_stop", "answer": None, "truth": 1}
    assert r["c_closed_hi"]["first_failure"]["answer"] == 0 and r["c_closed_hi"]["first_failure"]["truth"] == 1
    # c 는 '모두보다 큰 x' 에서만 틀린다: 틀린 경우의 정답은 항상 len(a)
    for a in bs.sorted_arrays(5):
        for x in bs.targets():
            out = bs.run_variant("c_closed_hi", a, x)
            assert (out["outcome"] == "wrong") == (len(a) > 0 and x > a[-1])


def test_bound_is_tight():
    for n in list(range(1, 65)) + [1000]:
        a = list(range(n))
        worst = 0
        for x in range(-1, n + 1):
            c = bs.Counter()
            bs.lower_bound(a, x, c)
            worst = max(worst, c.steps)
        assert worst == bs.max_steps_bound(n) == math.ceil(math.log2(n + 1))


def test_cpython_bisect_handles_huge_range_now():
    big = sys.maxsize
    assert bisect.bisect(range(big), big - 3) == big - 2


def test_sampling_linear_count_equals_bisect_right():
    rng = np.random.default_rng(0)
    for V in (1, 2, 5, 48):
        p = rng.random(V)
        p /= p.sum()
        cdf = np.cumsum(p)
        u = rng.random(500) * cdf[-1]
        lin = np.array([bs.sample_index_linear(cdf, ui) for ui in u])
        assert np.array_equal(lin, bs.sample_index_bisect(cdf, u))
        assert np.array_equal(lin, np.array([bisect.bisect_right(list(cdf), ui) for ui in u]))
