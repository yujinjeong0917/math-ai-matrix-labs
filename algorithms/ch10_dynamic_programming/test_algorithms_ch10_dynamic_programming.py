"""알고리즘 10장 검증. `uv run pytest -q algorithms/ch10_dynamic_programming` 로 실행한다.

성질 기반 테스트는 Hypothesis 대신 시드를 고정한 무작위 입력으로 확인한다(저장소 의존성은 numpy, torch, pytest 뿐).
"""

import itertools
import random
import sys

import numpy as np
import pytest

import algorithms_ch10_dp as dp


def _pairs(seed=0, count=200, max_len=7, alpha="ACGT"):
    rng = random.Random(seed)
    # 빈 문자열, 한 글자, 본문 예시 같은 경계 사례를 먼저 넣는다. 무작위 쌍은 알파벳 4개, 길이 0..7.
    out = [("", ""), ("", "AC"), ("GT", ""), ("A", "A"), ("A", "C"), ("RUN", "RAIN"), ("PYTHON", "TYPHOON")]
    for _ in range(count):
        a = "".join(rng.choice(alpha) for _ in range(rng.randint(0, max_len)))
        b = "".join(rng.choice(alpha) for _ in range(rng.randint(0, max_len)))
        out.append((a, b))
    return out


def test_first_screen_numbers():
    # visits 리스트의 인덱스 k = F(k) 를 부른 횟수. F(0), F(1), ... F(5) 순서.
    # fib(5): 호출 15번, 서로 다른 부분 문제 6개, fib(2) 는 3번, fib(1) 은 5번 불린다
    visits = [0] * 6
    c = dp.Counter()
    assert dp.fib_naive(5, c, visits) == 5 and c.calls == 15
    assert visits == [3, 5, 3, 2, 1, 1]
    c = dp.Counter()
    dp.fib_naive(30, c)
    # fib(30) 호출 수 = 2 F(31) - 1 = 2 x 1346269 - 1. 정수 연산이라 정확한 등호.
    assert c.calls == 2_692_537 == 2 * dp.fib_value(31) - 1
    # RUN -> RAIN 표: 손으로 채운 값과 같다
    D = dp.edit_distance_table("RUN", "RAIN")
    # 행 = RUN 의 접두사(빈, R, RU, RUN), 열 = RAIN 의 접두사(빈, R, RA, RAI, RAIN). 마지막 칸 2 = A 넣기 + U 를 I 로 바꾸기(R, N 은 일치).
    assert D.tolist() == [[0, 1, 2, 3, 4], [1, 0, 1, 2, 3], [2, 1, 1, 2, 3], [3, 2, 2, 2, 2]]
    # 마지막 칸: min(위 3+1, 왼쪽 2+1, 대각선 2+0) = 2 (N 과 N 이 같아서 대각선은 +0)
    assert min(D[2, 4] + 1, D[3, 3] + 1, D[2, 3] + 0) == 2
    ops = dp.backtrace(D, "RUN", "RAIN")
    # R 일치, A 넣기, U->I 바꾸기, N 일치. 대각선 우선 규칙 때문에 이 순서로 나온다(비용 2 인 다른 순서도 있을 수 있음).
    assert [o[0] for o in ops] == ["match", "ins", "sub", "match"]
    assert dp.apply_ops("RUN", "RAIN", ops) == "RAIN"
    # 첫 화면 "15번 중 9번은 다시 푼 것": 6개 질문의 첫 호출을 빼면 9
    assert sum(v - 1 for v in visits) == 9


def test_first_screen_speedup_claim():
    # 첫 화면 "수십만 배": results/ch10.json 의 가장 큰 재귀/표 시간 비가 10만~100만 사이
    import json
    from pathlib import Path
    # 이 테스트는 실험을 돌려 저장한 results/ch10.json 의 측정 시간에 의존한다. 기기가 다르면 비율이 달라질 수 있는 취약한 검사(확인 필요).
    d = json.loads((Path(__file__).parent / "results" / "ch10.json").read_text())
    r12 = [r for r in d["E2_edit_small"]["rows"] if r["m"] == 12][0]
    f35 = [r for r in d["E1_fib"]["rows"] if r["n"] == 35][0]
    for ratio in (r12["naive_ms_median"] / r12["table_ms_median"], f35["naive_ms"] / f35["table_ms"]):
        assert 1e5 <= ratio < 1e6


def test_fib_call_formula_and_agreement():
    for n in range(0, 26):
        c = dp.Counter()
        v = dp.fib_naive(n, c)
        assert c.calls == 2 * dp.fib_value(n + 1) - 1
        assert v == dp.fib_memo(n) == dp.fib_table(n) == dp.fib_value(n)
    cm = dp.Counter()
    dp.fib_memo(30, cm)
    # (호출 수는 결정적이라 정확한 등호로 비교한다.)
    assert cm.calls == 2 * 30 - 1          # 처음 계산 31번 + 적어 둔 값을 바로 돌려준 28번
    ct = dp.Counter()
    dp.fib_table(30, ct)
    # 표 칸 F(0)..F(30) 31개.
    assert ct.cells == 31


def test_edit_distance_all_versions_agree():
    for a, b in _pairs():
        D = dp.edit_distance_table(a, b)
        d = int(D[-1, -1])
        assert dp.edit_distance_naive(a, b) == d
        assert dp.edit_distance_naive(a, b, shortcut=True) == d
        assert dp.edit_distance_memo(a, b) == d
        assert dp.edit_distance_two_rows(a, b) == d


def test_every_cell_is_prefix_distance():
    for a, b in _pairs(seed=1, count=40, max_len=6):
        D = dp.edit_distance_table(a, b)
        for i in range(len(a) + 1):
            for j in range(len(b) + 1):
                # 표의 모든 칸이 '접두사끼리의 편집거리'라는 정의와 맞는지 재귀로 독립 확인.
                assert D[i, j] == dp.edit_distance_naive(a[:i], b[:j], shortcut=True)


def test_metric_axioms_and_bounds():
    rng = random.Random(5)
    words = ["".join(rng.choice("AB") for _ in range(rng.randint(0, 5))) for _ in range(25)]
    # 거리의 공리: 자기 자신과 0, 대칭, 삼각 부등식(25 x 25 x 25 모든 조합 확인), 하한 |길이 차|, 상한 max(길이).
    d = {(x, y): dp.edit_distance_two_rows(x, y) for x in words for y in words}
    for x in words:
        assert d[(x, x)] == 0
        for y in words:
            assert d[(x, y)] == d[(y, x)]
            assert abs(len(x) - len(y)) <= d[(x, y)] <= max(len(x), len(y))
            if x != y:
                assert d[(x, y)] > 0
            for z in words:
                assert d[(x, z)] <= d[(x, y)] + d[(y, z)]


def test_backtrace_rebuilds_target_with_distance_cost():
    for a, b in _pairs(seed=2):
        D = dp.edit_distance_table(a, b)
        ops = dp.backtrace(D, a, b)
        assert dp.apply_ops(a, b, ops) == b
        # 일치가 아닌 연산(sub, del, ins)의 개수가 곧 편집거리.
        assert sum(op != "match" for op, _, _ in ops) == D[-1, -1]


def test_fill_order_respects_dependencies():
    order = []
    dp.edit_distance_table("PYTHON", "TYPHOON", None, order)
    # 칸이 채워진 순서 번호. 의존하는 세 칸(위, 왼쪽, 대각선)이 항상 먼저 채워지는지.
    pos = {cell: k for k, cell in enumerate(order)}
    for (i, j) in order:
        if i > 0 and j > 0:
            assert pos[(i - 1, j)] < pos[(i, j)]
            assert pos[(i, j - 1)] < pos[(i, j)]
            assert pos[(i - 1, j - 1)] < pos[(i, j)]


def test_naive_call_count_and_delannoy_visits():
    for m in range(1, 7):
        rng = random.Random(m)
        a = "".join(rng.choice("ACGT") for _ in range(m))
        b = "".join(rng.choice("ACGT") for _ in range(m))
        c = dp.Counter()
        visits = [[0] * (m + 1) for _ in range(m + 1)]
        dp.edit_distance_naive(a, b, c, visits)
        # 실제 재귀 호출 수가 점화식 C(i,j) 로 계산한 값과 일치.
        assert c.calls == dp.naive_call_count(m, m)
        for i in range(1, m + 1):
            for j in range(1, m + 1):
                # 안쪽 칸 (i, j) 가 불린 횟수 = Delannoy 수 D(m-i, m-j) (루트 (m, m) 에서 그 칸까지의 길 수).
                assert visits[i][j] == dp.delannoy(m - i, m - j)
    # 점화식으로 손계산한 값(m = 2 -> 19, m = 3 -> 94).
    assert dp.naive_call_count(2, 2) == 19 and dp.naive_call_count(3, 3) == 94
    cm = dp.Counter()
    dp.edit_distance_memo("ACGTAC", "GTCAGA", cm)
    # 6 글자 두 개의 표는 (6+1) x (6+1) = 49 칸. 메모 판이 모든 칸을 정확히 한 번 계산.
    assert cm.cells == 7 * 7


def _small_hmm(seed, K, M=3):
    nr = np.random.default_rng(seed)
    pi, A, B = dp.random_hmm(nr, K, M)
    return pi, A, B, nr


def test_viterbi_matches_bruteforce_and_op_count():
    for seed in range(12):
        for K in (2, 3):
            pi, A, B, nr = _small_hmm(seed, K)
            # 관측 길이 2..6. 전수 탐색이 K^T 라 작게 유지(최대 3^6 = 729).
            T = 2 + seed % 5
            obs = nr.integers(0, 3, size=T).tolist()
            lp, lA, lB = np.log(pi), np.log(A), np.log(B)
            cv, cb = dp.Counter(), dp.Counter()
            path, score = dp.viterbi(lp, lA, lB, obs, cv)
            bpath, bscore = dp.viterbi_bruteforce(lp, lA, lB, obs, cb)
            assert abs(score - bscore) < 1e-9
            assert abs(dp.path_logprob(lp, lA, lB, obs, path) - score) < 1e-9
            # 비터비 연산 수: 첫 시점 K + 이후 (T-1) x K^2.
            assert cv.ops == K + (T - 1) * K * K
            # 전수 탐색: 경로 K^T 개 x 항 2T 개.
            assert cb.ops == (K ** T) * 2 * T


def test_viterbi_handles_zero_probabilities():
    # pi 가 상태 0 만 허용하고 A 가 0<->1 로만 이동하게 만든 결정적 모형이라 답이 [0,1,0,1] 하나뿐.
    # 0 확률 전이는 log 에서 -inf 가 되고, 그 길은 고르지 않는다
    pi = np.array([1.0, 0.0])
    A = np.array([[0.0, 1.0], [1.0, 0.0]])
    B = np.array([[0.9, 0.1], [0.2, 0.8]])
    obs = [0, 1, 0, 1]
    path, _ = dp.viterbi(dp.log0(pi), dp.log0(A), dp.log0(B), obs)
    assert path == [0, 1, 0, 1]


def test_probability_space_underflows_but_log_space_does_not():
    pi, A, B, nr = _small_hmm(2026, 3, 4)
    # 길이 1500. 확률 공간에서는 어느 시점에 모든 delta 가 0.0 이 되고(언더플로), 로그 공간은 유한한 값이다.
    obs = nr.integers(0, 4, size=1500).tolist()
    _, pval, first_zero = dp.viterbi_prob(pi, A, B, obs)
    assert pval == 0.0 and first_zero is not None
    _, lval = dp.viterbi(np.log(pi), np.log(A), np.log(B), obs)
    assert np.isfinite(lval)
    # 언더플로 직전 길이까지는 두 판이 같은 경로를 낸다.
    short = obs[: first_zero - 1]
    assert dp.viterbi_prob(pi, A, B, short)[0] == dp.viterbi(np.log(pi), np.log(A), np.log(B), short)[0]


def test_coin_change_matches_bruteforce():
    for coins in ([1, 3, 4], [1, 5, 10, 25], [1, 7, 10], [2, 5]):
        best, _ = dp.coin_change_dp(coins, 25)
        for amount in range(0, 26):
            # [2, 5] 처럼 1원이 없는 체계는 불가능한 금액이 INF 로 같아야 한다.
            assert best[amount] == dp.coin_change_bruteforce(coins, amount)
    best, combo = dp.coin_change_dp([1, 3, 4], 6)
    assert best[6] == 2 and combo == [3, 3] and dp.greedy_coins([1, 3, 4], 6) == [4, 1, 1]


def test_memo_hits_recursion_limit_table_does_not():
    # 기본 재귀 한도(1000)에서만 의미가 있는 테스트라 한도를 바꿨으면 건너뛴다.
    if sys.getrecursionlimit() > 1000:
        pytest.skip("재귀 한도가 기본값이 아님")
    with pytest.raises(RecursionError):
        dp.fib_memo(2000)
    assert dp.fib_table(2000) == dp.fib_value(2000)
    # 메모 편집거리의 재귀 깊이는 m + n = 1200 이 되어 한도 1000 을 넘는다.
    a = "A" * 600
    with pytest.raises(RecursionError):
        dp.edit_distance_memo(a, a)
    assert dp.edit_distance_two_rows(a, a) == 0


def test_bruteforce_enumerates_all_paths():
    K, T = 3, 4
    # 3^4 = 81: 전수 탐색이 훑는 경로 수.
    assert len(list(itertools.product(range(K), repeat=T))) == 81
