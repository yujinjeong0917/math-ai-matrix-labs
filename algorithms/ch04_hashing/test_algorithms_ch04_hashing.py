"""알고리즘 4장 검증. `uv run pytest -q algorithms/ch04_hashing` 로 실행한다.

설계도는 Hypothesis 상태 기계(RuleBasedStateMachine)를 적었지만 의존성을 늘리지 않으려고, 시드를 고정한
무작위 삽입·삭제·조회 열을 dict 와 나란히 돌리는 반복문으로 같은 성질을 확인한다.
범용 해시의 충돌 확률은 표본 추정 대신 작은 소수에서 모든 (a, b) 와 모든 키 쌍을 세는 전수 검사로 확인한다.
"""

import random
import sys

import numpy as np

import algorithms_ch04_hash as hh


def test_first_screen_numbers():
    # 사물함 8칸: 섞인 키는 25 와 41 만 1번 칸에서 만나 비교 1번, 8의 배수 5개는 모두 0번 칸이라 0+1+2+3+4 = 10
    m = 8
    assert [x % m for x in (3, 12, 25, 30, 41)] == [3, 4, 1, 6, 1]
    t, c = hh.build([3, 12, 25, 30, 41], m, hh.mod_hash(m))
    assert c.comparisons == 1 and t.max_chain() == 2
    t, c = hh.build([8, 16, 24, 32, 40], m, hh.mod_hash(m))
    assert c.comparisons == 0 + 1 + 2 + 3 + 4 == 10 and t.max_chain() == 5
    # 범용 해시 한 번(p=43, a=6, b=1): (6x+1) mod 43 을 먼저 구하고 다시 mod 8
    u = hh.UniversalHash(43, m, 6, 1)
    assert [6 * x + 1 for x in (8, 16, 24, 32, 40)] == [49, 97, 145, 193, 241]
    assert [(6 * x + 1) % 43 for x in (8, 16, 24, 32, 40)] == [6, 11, 16, 21, 26]
    assert [u(x) for x in (8, 16, 24, 32, 40)] == [6, 3, 0, 5, 2]
    # a = 6, b = 1 을 아는 사람은 6번 칸으로 가는 키를 금방 찾는다
    assert [u(x) for x in (8, 12, 25, 38, 42)] == [6] * 5


def test_adversarial_is_exactly_quadratic():
    for n in (1, 2, 10, 500, 1500):
        t, c = hh.build(hh.adversarial_keys(1024, n), 1024, hh.mod_hash(1024))
        assert c.comparisons == n * (n - 1) // 2  # k 번째 키가 앞선 k-1 개와 비교: 합 n(n-1)/2
        assert t.max_chain() == n and t.chain_lengths()[0] == n


def test_table_matches_dict_random_ops():
    """삽입·덮어쓰기·삭제·조회를 섞은 열에서 dict 와 같은 답, 매 연산 뒤 불변식 성립, 크기 재조정 포함."""
    for trial in range(60):
        rng = random.Random(trial)
        use_uni = trial % 2 == 0  # 짝수 회는 범용 해시(p=97), 홀수 회는 mod 해시
        # 키는 0..96 (p=97 미만)이라 범용 해시 입력 범위를 지킨다. 30% 는 8의 배수로 골라 충돌을 일부러 늘린다
        m = rng.choice([1, 2, 7, 8, 16])
        h = hh.universal_hash(97, m, rng) if use_uni else hh.mod_hash(m)
        t = hh.ChainedHashTable(m, h, max_load=rng.choice([None, 0.75, 2.0]))
        ref = {}
        for _ in range(300):
            op = rng.random()
            k = rng.randrange(0, 97) if rng.random() < 0.7 else rng.choice([0, 8, 16, 24, 32])
            if op < 0.5:
                v = rng.randrange(1000)
                assert t.insert(k, v) == (k not in ref)
                ref[k] = v
            elif op < 0.75:
                assert t.delete(k) == (k in ref)
                ref.pop(k, None)
            else:
                assert t.get(k, "없음") == ref.get(k, "없음")
            assert t.n == len(ref)
            assert all(ok for _, ok, _ in t.check_invariants())
        assert sorted(t.items()) == sorted(ref.items())
        if t.max_load is not None:
            assert t.n <= t.max_load * t.m


def test_resize_keeps_invariants_and_cannot_fix_mod_hash():
    # 8 -> 16 -> ... -> 512 로 6번 두 배(n 이 m 을 넘을 때마다). 그래도 2^20 의 배수는 512 로 나눈 나머지가 모두 0
    keys = [i * 2 ** 20 for i in range(300)]
    t, c = hh.build(keys, 8, hh.mod_hash(8), max_load=1.0)
    assert t.m == 512 and c.resizes == 6
    assert all(ok for _, ok, _ in t.check_invariants())
    assert t.max_chain() == 300  # 2^20 의 배수는 m = 512 에서도 모두 0번 칸
    t2, _ = hh.build(keys, 8, hh.universal_hash(2 ** 31 - 1, 8, random.Random(0)), max_load=1.0)
    assert t2.max_chain() < 20


def test_carter_wegman_h1_universal_exhaustive():
    """Lemma 6 / Proposition 7: 모든 x != y 에 대해 충돌하는 (a, b) 의 수가 p(p-1)/m 이하이고, 그 수는 x, y 와 무관하다."""
    # 작은 소수 p 에서 모든 (a, b) 와 모든 키 쌍 (x, y) 를 세는 전수 검사(표본 추정이 아니다)
    for p, m in ((31, 4), (31, 8), (37, 5)):
        xs = np.arange(p, dtype=np.int64)
        coll = np.zeros((p, p), dtype=np.int64)
        for a in range(1, p):
            hv = ((a * xs[None, :] + np.arange(p, dtype=np.int64)[:, None]) % p) % m
            for row in hv:
                coll += row[:, None] == row[None, :]
        # off: 대각선(x == y)을 뺀 x != y 쌍들의 충돌 (a,b) 개수. 모든 쌍에서 같은 값이어야 한다
        off = coll[~np.eye(p, dtype=bool)]
        sizes = [len(range(r, p, m)) for r in range(m)]
        # sizes[r]: 0..p-1 중 나머지가 r 인 값의 수. 충돌 (a,b) 수는 이 집단 안의 서로 다른 값 순서쌍 수와 같다
        assert off.min() == off.max() == sum(s * (s - 1) for s in sizes)
        assert off.max() <= p * (p - 1) / m


def test_mod_hash_is_not_universal():
    # 고정 해시 하나뿐이면 0 과 m 은 "모든 함수"에서(=그 하나에서) 충돌한다: 확률 1
    for m in (2, 8, 1024):
        h = hh.mod_hash(m)
        assert h(0) == h(m) == h(5 * m)


def test_known_seed_attack_and_redraw():
    # a, b 가 공개된 해시에서는 같은 칸으로 가는 키 200개를 찾아 n(n-1)/2 비교를 유발할 수 있고,
    # 다른 (a, b) 로 바꾸면(공격자가 모르는 새 시드) 그 키 집합이 흩어진다
    h = hh.universal_hash(2 ** 31 - 1, 64, random.Random(3))
    keys, tries = hh.keys_for_known_seed(h, 200)
    assert len(set(h(k) for k in keys)) == 1
    t, c = hh.build(keys, 64, h)
    assert c.comparisons == 200 * 199 // 2
    h2 = hh.universal_hash(2 ** 31 - 1, 64, random.Random(4))
    t2, c2 = hh.build(keys, 64, h2)
    # 30 과 2000 은 이 시드에서 여유 있게 넘지 않는 값으로 잡은 경험적 경계(이론 보장은 아님)
    assert t2.max_chain() < 30 and c2.comparisons < 2000


def test_search_cost_exact_formula_for_one_table():
    """체인 끝에 붙이므로 i 번째로 넣은 키를 찾을 때 비교 수 = 1 + (그 칸에 먼저 들어간 키 수)."""
    rng = random.Random(5)
    keys = hh.random_keys(500, rng)
    h = hh.universal_hash(2 ** 31 - 1, 64, rng)
    t, _ = hh.build(keys, 64, h)
    seen = {}
    expected = 0
    for k in keys:
        b = h(k)
        expected += 1 + seen.get(b, 0)
        seen[b] = seen.get(b, 0) + 1
    assert hh.successful_search_cost(t, keys) * len(keys) == expected
    # 평균으로 바꾸면 sum(L_b(L_b+1)/2)/n 이다
    lens = t.chain_lengths()
    assert expected == sum(L * (L + 1) // 2 for L in lens)


def test_numpy_vectorized_hash_matches_table():
    h = hh.universal_hash(2 ** 31 - 1, 256, random.Random(9))
    keys = hh.random_keys(3000, random.Random(9))
    t, _ = hh.build(keys, 256, h)
    # 같은 식을 NumPy 로 한꺼번에 계산. a*x 는 최대 약 2^62 이라 int64 로 넘치지 않는다(a, x < 2^31)
    vec = ((h.a * np.array(keys, dtype=np.int64) + h.b) % (2 ** 31 - 1)) % 256
    assert np.bincount(vec, minlength=256).tolist() == t.chain_lengths()
    assert [h(k) for k in keys[:50]] == vec[:50].tolist()


def test_python_int_hash_is_not_salted():
    # 파이썬 int 의 hash 는 x mod (2^61 - 1) 이라(64비트 기준) 소금이 없다. str 과 다르다
    P = sys.hash_info.modulus
    assert all(hash(k * P) == 0 for k in range(1, 50))
    assert hash(12345) == 12345
    assert sys.hash_info.algorithm in ("siphash13", "siphash24", "fnv")


def test_universal_hash_rejects_out_of_range_keys():
    u = hh.UniversalHash(43, 8, 6, 1)
    try:
        u(43)
    except AssertionError:
        return
    raise AssertionError("p 이상의 키를 받아들였다")
