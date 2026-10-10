"""알고리즘 5장 검증. `uv run pytest -q algorithms/ch05_balanced_tree` 로 실행한다.

설계도는 Hypothesis 성질 테스트를 적었지만 의존성을 늘리지 않으려고, 시드를 고정한 무작위 넣기·지우기·찾기 열을
정렬된 집합(set + sorted)과 나란히 돌리는 반복문으로 같은 성질을 확인한다.
"""

import heapq
import itertools
import math
import random
import sys

import numpy as np
import pytest

import algorithms_ch05_trees as T

C_PHI = 1 / math.log2((1 + 5 ** 0.5) / 2)


def test_first_screen_numbers():
    t, c = T.build("bst", range(1, 8))
    assert T.height(t.root) == 7 and c.comparisons == 0 + 1 + 2 + 3 + 4 + 5 + 6 == 21
    assert sum(T.depths(t.root).values()) == 28  # 1+2+...+7, 평균 4
    t, c = T.build("avl", range(1, 8))
    assert T.height(t.root) == 3 and c.rotations == 4 and c.comparisons == 14
    assert sum(T.depths(t.root).values()) == 1 + 2 * 2 + 4 * 3 == 17
    t, c = T.build("bst", [4, 2, 6, 1, 3, 5, 7])
    assert T.height(t.root) == 3 and c.comparisons == 0 + 1 + 1 + 2 + 2 + 2 + 2 == 10


def test_sorted_and_zigzag_make_a_chain():
    for order in ("sorted", "zigzag"):
        for n in (1, 2, 10, 300):
            t, c = T.build("bst", T.orders(order, n))
            assert T.height(t.root) == n
            assert c.comparisons == n * (n - 1) // 2
            assert T.mean_search_comparisons(t.root, range(1, n + 1)) == (n + 1) / 2


def test_rotations_keep_inorder():
    rng = random.Random(0)
    for _ in range(200):
        keys = rng.sample(range(100), rng.randint(3, 30))
        t, _ = T.build("bst", keys)  # 아무 모양의 BST
        before = T.inorder(t.root)
        for rot in (T.rotate_left, T.rotate_right):
            root = t.root
            child = root.right if rot is T.rotate_left else root.left
            if child is None:
                continue
            new = rot(root)
            assert T.inorder(new) == before
            # 되돌리기
            t.root = (T.rotate_right if rot is T.rotate_left else T.rotate_left)(new)
            assert T.inorder(t.root) == before


def test_avl_matches_sorted_set_random_ops():
    """넣기·지우기·찾기를 섞은 열에서 정렬된 집합과 같은 답, 매 연산 뒤 불변식 세 개, 높이 상한."""
    for trial in range(40):
        rng = random.Random(trial)
        t, ref = T.AVL(), set()
        universe = rng.choice([10, 50, 500])
        for _ in range(400):
            k = rng.randrange(universe)
            op = rng.random()
            if op < 0.5:
                assert t.insert(k) == (k not in ref)
                ref.add(k)
            elif op < 0.8:
                assert t.delete(k) == (k in ref)
                ref.discard(k)
            else:
                assert t.search(k) == (k in ref)
            assert t.n == len(ref)
            assert all(ok for _, ok, _ in T.check_avl(t.root))
            if t.n:
                assert T.height(t.root) <= C_PHI * math.log2(t.n + 1) + 1e-9
        assert T.inorder(t.root) == sorted(ref)


def test_bst_delete_matches_sorted_set():
    for trial in range(30):
        rng = random.Random(1000 + trial)
        t, ref = T.BST(), set()
        for _ in range(300):
            k = rng.randrange(60)
            if rng.random() < 0.6:
                assert t.insert(k) == (k not in ref)
                ref.add(k)
            else:
                assert t.delete(k) == (k in ref)
                ref.discard(k)
            assert T.check_bst_order(t.root)[1]
        assert T.inorder(t.root) == sorted(ref)


def test_min_avl_nodes_is_fibonacci_minus_one():
    fib = [0, 1]
    for _ in range(40):
        fib.append(fib[-1] + fib[-2])
    phi = (1 + 5 ** 0.5) / 2
    for h in range(1, 30):
        N = T.min_avl_nodes(h)
        assert N == T.min_avl_nodes(h - 1) + (T.min_avl_nodes(h - 2) if h >= 2 else 0) + 1
        assert N == fib[h + 2] - 1
        assert N + 1 >= phi ** h - 1e-9  # 그래서 h <= log_phi(N+1)
        assert h <= C_PHI * math.log2(N + 1) + 1e-9 <= 1.5 * math.log2(N + 1)


def test_fibonacci_tree_is_thinnest_avl():
    for h in range(1, 12):
        order, root = T.fibonacci_tree_keys(h)
        assert len(order) == T.min_avl_nodes(h) and T.height(root) == h
        assert all(ok for _, ok, _ in T.check_avl(root))
        # 키 하나만 빼도(가장 깊은 잎) 높이 h 인 AVL 이 아니게 된다는 건 N(h) 정의에서 나온다. 여기서는 모양만 확인
        bst, _ = T.build("bst", order)
        assert T.snapshot(bst.root)[1] == T.snapshot(root)[1]


def test_exhaustive_avl_height_bound_small_n():
    """n <= 7 인 모든 넣는 순서에서 AVL 높이는 ceil(log2(n+1)) 이상, 상한 이하."""
    for n in range(1, 8):
        heights = set()
        for perm in itertools.permutations(range(n)):
            t, _ = T.build("avl", perm)
            assert all(ok for _, ok, _ in T.check_avl(t.root))
            heights.add(T.height(t.root))
        assert min(heights) >= math.ceil(math.log2(n + 1))
        assert max(heights) <= C_PHI * math.log2(n + 1) + 1e-9


def test_recursive_insert_hits_recursion_limit_on_sorted_input():
    t = T.BST()
    with pytest.raises(RecursionError):
        for k in range(1, sys.getrecursionlimit() + 10):
            t.insert_recursive(k)
    # 반복판은 같은 입력을 넣는다
    t2, _ = T.build("bst", range(1, sys.getrecursionlimit() + 10))
    assert T.height(t2.root) == sys.getrecursionlimit() + 9


def test_heap_matches_heapq_and_sorted():
    for trial in range(30):
        rng = random.Random(trial)
        xs = [rng.randrange(50) for _ in range(rng.randint(0, 200))]
        h1 = T.BinaryHeap()
        h1.heapify(xs)
        h2 = T.BinaryHeap()
        for x in xs:
            h2.push(x)
        for h in (h1, h2):
            assert T.check_heap(h.a)[1]
            a = np.array(h.a, dtype=np.int64)
            idx = np.arange(1, len(a))
            assert (a[(idx - 1) // 2] <= a[idx]).all()
        ref = list(xs)
        heapq.heapify(ref)
        assert [h1.pop() for _ in range(len(xs))] == [heapq.heappop(ref) for _ in range(len(xs))] == sorted(xs)
        assert [h2.pop() for _ in range(len(xs))] == sorted(xs)


def test_heapify_is_linear_push_is_not():
    for n in (100, 1000, 5000):
        xs = list(range(n, 0, -1))  # 최소 힙에 내림차순: push 는 매번 뿌리까지 올라간다
        c1, c2 = T.Counter(), T.Counter()
        T.BinaryHeap(c1).heapify(xs)
        h2 = T.BinaryHeap(c2)
        for x in xs:
            h2.push(x)
        assert c1.sift <= 2 * n
        # push i 번째(0부터)는 깊이 floor(log2(i+1)) 만큼 비교한다
        assert c2.sift == sum(int(math.log2(i + 1)) for i in range(n))


def test_trace_widget_shape():
    tr = T.trace_inserts(list(range(1, 8)), "avl")
    kinds = [s[9] for s in tr["steps"]]
    assert kinds.count("violation") == kinds.count("rotated") == 4
    for s in tr["steps"]:
        if s[9] == "violation":
            assert s[7]  # 깨진 노드가 있다
        else:
            assert not s[7] and s[8]
    tb = T.trace_inserts(list(range(1, 8)), "bst")
    assert tb["height"] == 7 and all(s[9] in ("start", "insert") for s in tb["steps"])
