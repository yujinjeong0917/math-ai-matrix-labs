"""알고리즘 4장 최소 구현: 체이닝 해시 테이블, 고정 해시(x mod m), Carter-Wegman(1979)의 범용 해시족 H1.

표준 라이브러리만 쓴다(NumPy 대조는 experiments.py 와 테스트에서만).

셈 규칙은 하나다: 키 두 개를 `==` 로 한 번 비교할 때마다 comparisons 를 1 올린다.
삽입은 "이미 있는 키인지" 체인을 끝까지 훑은 뒤 없으면 체인 끝에 붙인다. 그래서 한 칸에 몰린
k 번째 키를 넣을 때 k-1 번 비교하고, n 개가 모두 한 칸에 몰리면 삽입 전체가 정확히 n(n-1)/2 번이다.

들어 있는 것
    Counter               : comparisons(키 비교), hash_calls(해시 계산), resizes(크기 재조정 횟수)
    ModHash / mod_hash(m) : h(x) = x mod m. 입력에 무작위가 없다.
    UniversalHash / universal_hash(p, m, rng)
                          : h(x) = ((a x + b) mod p) mod m, a 는 1..p-1, b 는 0..p-1 에서 실행할 때 뽑는다.
                            Carter·Wegman(1979) 의 H1 이며, 키는 0..p-1 이어야 한다.
    ChainedHashTable      : insert / get / delete / resize, 불변식 검사 check_invariants
    adversarial_keys(m, n): m 의 배수 0, m, 2m, ... (x mod m 이 모두 0)
    keys_for_known_seed   : a, b 를 아는 공격자가 한 칸에 몰리는 키를 무차별 대입으로 찾는다.
"""

import random


class Counter:
    def __init__(self):
        self.comparisons = 0
        self.hash_calls = 0
        self.resizes = 0


def _c(counter):
    return counter if counter is not None else Counter()


# ---------------------------------------------------------------- 해시 함수

class ModHash:
    """h(x) = x mod m. 키를 고르는 사람이 m 을 알면 모든 키를 한 칸에 넣을 수 있다."""

    def __init__(self, m):
        self.m = m

    def __call__(self, x):
        return x % self.m

    def with_m(self, m):
        return ModHash(m)

    def __repr__(self):
        return f"ModHash(m={self.m})"


def mod_hash(m):
    return ModHash(m)


class UniversalHash:
    """Carter·Wegman(1979) H1: f_{a,b}(x) = ((a x + b) mod p) mod m, a != 0.

    서로 다른 두 키 x, y 에 대해, (a, b) 를 고르게 뽑으면 Pr[f(x) = f(y)] <= 1/m 이다(논문 Proposition 7).
    키는 0 <= x < p 여야 한다. 크기를 바꿀 때(with_m) a, b 는 그대로 두고 m 만 바꾼다.
    """

    def __init__(self, p, m, a, b):
        assert 1 <= a < p and 0 <= b < p
        self.p, self.m, self.a, self.b = p, m, a, b

    def __call__(self, x):
        assert 0 <= x < self.p, "범용 해시 H1 의 키는 0..p-1 이어야 한다"
        return ((self.a * x + self.b) % self.p) % self.m

    def with_m(self, m):
        return UniversalHash(self.p, m, self.a, self.b)

    def __repr__(self):
        return f"UniversalHash(p={self.p}, m={self.m}, a={self.a}, b={self.b})"


def universal_hash(p, m, rng):
    """rng(random.Random)로 a, b 를 뽑는다. 공격자는 이 rng 의 상태를 모른다는 게 전제다."""
    a = rng.randrange(1, p)
    b = rng.randrange(0, p)
    return UniversalHash(p, m, a, b)


P31 = 2 ** 31 - 1  # 메르센 소수. Carter·Wegman 이 mod p 를 나눗셈 없이 하는 예로 든 2^j - 1 꼴


# ---------------------------------------------------------------- 체이닝 해시 테이블

class ChainedHashTable:
    """버킷 m 개, 버킷마다 [키, 값] 목록(체인). max_load 를 주면 n/m 이 그것을 넘을 때 m 을 두 배로 늘린다."""

    def __init__(self, m, hash_fn, counter=None, max_load=None):
        assert hash_fn.m == m
        self.m = m
        self.h = hash_fn
        self.c = _c(counter)
        self.max_load = max_load
        self.n = 0
        self.buckets = [[] for _ in range(m)]

    def _bucket(self, key):
        self.c.hash_calls += 1
        return self.buckets[self.h(key)]

    def _find(self, chain, key):
        for i, kv in enumerate(chain):
            self.c.comparisons += 1
            if kv[0] == key:
                return i
        return -1

    def insert(self, key, value=None):
        """새 키면 True, 이미 있던 키면 값만 바꾸고 False."""
        chain = self._bucket(key)
        i = self._find(chain, key)
        if i >= 0:
            chain[i][1] = value
            return False
        chain.append([key, value])
        self.n += 1
        if self.max_load is not None and self.n > self.max_load * self.m:
            self.resize(2 * self.m)
        return True

    def get(self, key, default=None):
        chain = self._bucket(key)
        i = self._find(chain, key)
        return chain[i][1] if i >= 0 else default

    def contains(self, key):
        chain = self._bucket(key)
        return self._find(chain, key) >= 0

    def delete(self, key):
        chain = self._bucket(key)
        i = self._find(chain, key)
        if i < 0:
            return False
        chain.pop(i)
        self.n -= 1
        return True

    def resize(self, new_m):
        """버킷 수를 new_m 으로 바꾸고 모든 키를 새 h 로 다시 넣는다(이미 서로 다른 키라 비교는 하지 않는다)."""
        old = [kv for chain in self.buckets for kv in chain]
        self.m = new_m
        self.h = self.h.with_m(new_m)
        self.buckets = [[] for _ in range(new_m)]
        for kv in old:
            self.c.hash_calls += 1
            self.buckets[self.h(kv[0])].append(kv)
        self.c.resizes += 1

    # ---- 관찰용
    def chain_lengths(self):
        return [len(ch) for ch in self.buckets]

    def max_chain(self):
        return max(self.chain_lengths()) if self.m else 0

    def items(self):
        return [(k, v) for ch in self.buckets for k, v in ch]

    def check_invariants(self):
        """[(이름, 성립 여부, 설명)]. 해시 계산은 계수기에 세지 않는다."""
        lens = self.chain_lengths()
        placed = all(self.h(k) == b for b, ch in enumerate(self.buckets) for k, _ in ch)
        keys = [k for ch in self.buckets for k, _ in ch]
        return [
            ("모든 키는 h(키) 버킷에 있다", placed, ""),
            ("체인 길이의 합 = n", sum(lens) == self.n, f"{sum(lens)} vs {self.n}"),
            ("같은 키는 한 번만 들어 있다", len(set(keys)) == len(keys), ""),
        ]


# ---------------------------------------------------------------- 입력 만들기

def adversarial_keys(m, n):
    """x mod m 이 모두 0 인 키 n 개."""
    return [i * m for i in range(n)]


def random_keys(n, rng, hi=P31):
    """0..hi-1 에서 서로 다른 키 n 개."""
    return rng.sample(range(hi), n)


def keys_for_known_seed(h, n, target=None, start=0):
    """해시 h(a, b, p, m 이 모두 공개된 경우)에서 같은 칸(target, 기본은 h(start))으로 가는 키 n 개를
    start 부터 하나씩 시험해 찾는다. (키 목록, 시험한 횟수)를 돌려준다. 기대 시험 횟수는 약 n*m 이다."""
    if target is None:
        target = h(start)
    out, x, tries = [], start, 0
    while len(out) < n:
        tries += 1
        if h(x) == target:
            out.append(x)
        x += 1
    return out, tries


# ---------------------------------------------------------------- 한꺼번에 넣기

def build(keys, m, hash_fn, max_load=None):
    c = Counter()
    t = ChainedHashTable(m, hash_fn, c, max_load=max_load)
    for k in keys:
        t.insert(k)
    return t, c


def successful_search_cost(t, keys):
    """keys 각각을 찾을 때의 비교 수 평균."""
    before = t.c.comparisons
    for k in keys:
        assert t.contains(k)
    return (t.c.comparisons - before) / len(keys)


def unsuccessful_search_cost(t, keys):
    before = t.c.comparisons
    for k in keys:
        assert not t.contains(k)
    return (t.c.comparisons - before) / len(keys)


# ---------------------------------------------------------------- 위젯 트레이스

def trace_inserts(keys, hash_fn):
    """삽입 한 번마다 한 단계. [설명, 키, 버킷, 체인 길이 목록, 버킷 내용, 누적 비교, [불변식1, 불변식2]]."""
    c = Counter()
    t = ChainedHashTable(hash_fn.m, hash_fn, c)
    steps = [["start", None, None, t.chain_lengths(), [[] for _ in range(t.m)], 0, [True, True]]]
    for k in keys:
        b = hash_fn(k)
        before = c.comparisons
        t.insert(k)
        inv = t.check_invariants()
        steps.append([f"키 {k} → {b}번 칸, 비교 {c.comparisons - before}번", k, b, t.chain_lengths(),
                      [[kk for kk, _ in ch] for ch in t.buckets], c.comparisons, [inv[0][1], inv[1][1]]])
    return {"keys": list(keys), "m": hash_fn.m, "hash": repr(hash_fn), "comparisons": c.comparisons,
            "max_chain": t.max_chain(), "steps": steps}


def demo_rng(seed):
    return random.Random(seed)
