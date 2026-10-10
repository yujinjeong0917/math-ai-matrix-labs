"""알고리즘 5장 최소 구현: 균형 장치 없는 이진 탐색 트리(BST), AVL 트리(Adelson-Velsky·Landis 1962), 이진 힙.

표준 라이브러리만 쓴다(NumPy 대조는 experiments.py 와 테스트에서만).

셈 규칙
    comparisons : 찾는 키를 노드의 키와 한 번 견줄 때마다 1. AVL 논문처럼 한 번의 비교가 "작다/같다/크다"
                  세 답을 낸다고 본다. 그래서 깊이 d(뿌리 = 1)에 있는 키를 찾으면 정확히 d 번이다.
    rotations   : 회전 한 번마다 1. 이중 회전은 2로 센다(single, double 은 따로도 센다).
    높이         : 뿌리에서 가장 먼 노드까지 경로에 있는 노드 수. 빈 트리 0, 노드 하나 1.
                  AVL 논문(1962)의 "가지 길이"는 한 셀 아래에 매달린 셀만 세서 노드 하나가 0 이다.
                  그래서 논문의 n 은 이 파일의 h - 1 이고, 논문 Lemma 1 은 여기 단위로 h - 1 < log_phi(N+1) 이다.

들어 있는 것
    Counter                         : comparisons, rotations, single, double, sift(힙 비교)
    BST                             : insert(반복판), insert_recursive(재귀판, RecursionError 재현용), search, delete
    AVL                             : insert, delete, search, rotate_left, rotate_right, rebalance
    height, inorder, depths         : 관찰용
    check_bst_order, check_avl      : 불변식 검사 [(이름, 성립 여부, 설명)]
    min_avl_nodes(h)                : 높이 h 인 AVL 트리의 최소 노드 수 N(h) = N(h-1) + N(h-2) + 1
    BinaryHeap, check_heap          : 최소 힙. push / pop / heapify(Floyd 식 아래에서 위로)
    orders(name, n, rng)            : 넣는 순서 3종(sorted, random, zigzag)
    trace_inserts(keys, kind)       : 위젯용 단계 기록
"""

import random


class Counter:
    def __init__(self):
        self.comparisons = 0
        self.rotations = 0
        self.single = 0
        self.double = 0
        self.sift = 0


def _c(counter):
    return counter if counter is not None else Counter()


class Node:
    __slots__ = ("key", "left", "right", "h")

    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.h = 1  # AVL 만 이 값을 유지한다. BST 는 쓰지 않는다


# ---------------------------------------------------------------- 관찰용 함수

def height(node):
    """저장된 값이 아니라 구조에서 다시 센 높이. 깊은 트리에서도 넘치지 않게 반복으로 센다."""
    best, stack = 0, [(node, 1)] if node else []
    while stack:
        nd, d = stack.pop()
        best = max(best, d)
        if nd.left:
            stack.append((nd.left, d + 1))
        if nd.right:
            stack.append((nd.right, d + 1))
    return best


def inorder(node):
    out, stack, cur = [], [], node
    while stack or cur:
        while cur:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        out.append(cur.key)
        cur = cur.right
    return out


def depths(node):
    """{키: 깊이(뿌리 1)}. 깊이 = 그 키를 찾을 때의 비교 수."""
    out, stack = {}, [(node, 1)] if node else []
    while stack:
        nd, d = stack.pop()
        out[nd.key] = d
        if nd.left:
            stack.append((nd.left, d + 1))
        if nd.right:
            stack.append((nd.right, d + 1))
    return out


def size(node):
    return len(depths(node))


def _heights_postorder(node):
    """{id(노드): 구조에서 센 높이}. 반복 후위 순회."""
    hs, stack = {}, [(node, False)] if node else []
    while stack:
        nd, done = stack.pop()
        if done:
            hl = hs[id(nd.left)] if nd.left else 0
            hr = hs[id(nd.right)] if nd.right else 0
            hs[id(nd)] = 1 + max(hl, hr)
        else:
            stack.append((nd, True))
            if nd.left:
                stack.append((nd.left, False))
            if nd.right:
                stack.append((nd.right, False))
    return hs


def balance_factors(node):
    """{키: 왼쪽 높이 - 오른쪽 높이}. 저장된 h 를 믿지 않고 구조에서 센다."""
    hs = _heights_postorder(node)
    out = {}
    for nd in _nodes(node):
        hl = hs[id(nd.left)] if nd.left else 0
        hr = hs[id(nd.right)] if nd.right else 0
        out[nd.key] = hl - hr
    return out


def _nodes(node):
    out, stack = [], [node] if node else []
    while stack:
        nd = stack.pop()
        out.append(nd)
        if nd.left:
            stack.append(nd.left)
        if nd.right:
            stack.append(nd.right)
    return out


def check_bst_order(root):
    keys = inorder(root)
    ok = all(a < b for a, b in zip(keys, keys[1:]))
    return ("중위 순회가 정렬돼 있다", ok, "")


def check_avl(root):
    """[(이름, 성립 여부, 설명)]. 균형 인수와 저장된 높이를 구조와 대조한다."""
    bfs = balance_factors(root)
    bad = sorted(k for k, b in bfs.items() if abs(b) > 1)
    hs = _heights_postorder(root)
    stored_ok = all(nd.h == hs[id(nd)] for nd in _nodes(root))
    return [
        check_bst_order(root),
        ("모든 노드의 균형 인수가 -1, 0, 1 중 하나다", not bad, f"깨진 노드 {bad}" if bad else ""),
        ("저장된 높이가 실제 높이와 같다", stored_ok, ""),
    ]


# ---------------------------------------------------------------- 균형 장치 없는 BST

class BST:
    def __init__(self, counter=None):
        self.root = None
        self.n = 0
        self.c = _c(counter)

    def insert(self, key):
        """반복판. 새 키면 True. 깊이가 n 이 돼도 파이썬 스택을 쓰지 않는다."""
        if self.root is None:
            self.root = Node(key)
            self.n = 1
            return True
        cur = self.root
        while True:
            self.c.comparisons += 1
            if key == cur.key:
                return False
            if key < cur.key:
                if cur.left is None:
                    cur.left = Node(key)
                    break
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = Node(key)
                    break
                cur = cur.right
        self.n += 1
        return True

    def insert_recursive(self, key):
        """교과서식 재귀판. 깊이가 파이썬 재귀 한도에 닿으면 RecursionError 가 난다."""
        def rec(node):
            if node is None:
                self.n += 1
                return Node(key)
            self.c.comparisons += 1
            if key < node.key:
                node.left = rec(node.left)
            elif key > node.key:
                node.right = rec(node.right)
            return node
        self.root = rec(self.root)

    def search(self, key):
        cur = self.root
        while cur:
            self.c.comparisons += 1
            if key == cur.key:
                return True
            cur = cur.left if key < cur.key else cur.right
        return False

    def delete(self, key):
        """Hibbard 식 삭제: 자식이 둘이면 오른쪽 부분 트리의 최솟값으로 바꾼다. 반복판."""
        parent, cur = None, self.root
        while cur and cur.key != key:
            parent, cur = cur, (cur.left if key < cur.key else cur.right)
        if cur is None:
            return False
        if cur.left and cur.right:
            sp, s = cur, cur.right
            while s.left:
                sp, s = s, s.left
            cur.key = s.key
            parent, cur = sp, s
        child = cur.left or cur.right
        if parent is None:
            self.root = child
        elif parent.left is cur:
            parent.left = child
        else:
            parent.right = child
        self.n -= 1
        return True


# ---------------------------------------------------------------- AVL 트리

def _h(node):
    return node.h if node else 0


def _fix(node):
    node.h = 1 + max(_h(node.left), _h(node.right))


def rotate_right(y):
    """왼쪽으로 기운 y 를 오른쪽으로 돌린다. 중위 순서(A x B y C)는 그대로다.

          y            x
         / \\          / \\
        x   C   ->   A   y
       / \\              / \\
      A   B            B   C
    """
    x = y.left
    y.left = x.right
    x.right = y
    _fix(y)
    _fix(x)
    return x


def rotate_left(x):
    y = x.right
    x.right = y.left
    y.left = x
    _fix(x)
    _fix(y)
    return y


class AVL:
    """모든 노드에서 |왼쪽 높이 - 오른쪽 높이| <= 1 을 지킨다. hook(이름, 노드)는 위젯 기록용."""

    def __init__(self, counter=None, hook=None):
        self.root = None
        self.n = 0
        self.c = _c(counter)
        self.hook = hook

    def rebalance(self, node):
        _fix(node)
        bf = _h(node.left) - _h(node.right)
        if -1 <= bf <= 1:
            return node
        if self.hook:
            self.hook("violation", node)
        if bf > 1:
            if _h(node.left.left) >= _h(node.left.right):
                self.c.rotations += 1
                self.c.single += 1
                kind = "오른쪽 회전 1번"
                out = rotate_right(node)
            else:
                self.c.rotations += 2
                self.c.double += 1
                kind = "왼쪽-오른쪽 이중 회전"
                node.left = rotate_left(node.left)
                out = rotate_right(node)
        else:
            if _h(node.right.right) >= _h(node.right.left):
                self.c.rotations += 1
                self.c.single += 1
                kind = "왼쪽 회전 1번"
                out = rotate_left(node)
            else:
                self.c.rotations += 2
                self.c.double += 1
                kind = "오른쪽-왼쪽 이중 회전"
                node.right = rotate_right(node.right)
                out = rotate_left(node)
        if self.hook:
            self.hook("rotated:" + kind, out)
        return out

    def insert(self, key):
        added = [False]

        def rec(node):
            if node is None:
                added[0] = True
                return Node(key)
            self.c.comparisons += 1
            if key == node.key:
                return node
            if key < node.key:
                node.left = rec(node.left)
            else:
                node.right = rec(node.right)
            return self.rebalance(node)

        self.root = rec(self.root)
        if added[0]:
            self.n += 1
        return added[0]

    def delete(self, key):
        removed = [False]

        def pop_min(node):
            if node.left is None:
                return node.right, node
            node.left, m = pop_min(node.left)
            return self.rebalance(node), m

        def rec(node):
            if node is None:
                return None
            self.c.comparisons += 1
            if key < node.key:
                node.left = rec(node.left)
            elif key > node.key:
                node.right = rec(node.right)
            else:
                removed[0] = True
                if node.left is None:
                    return node.right
                if node.right is None:
                    return node.left
                node.right, m = pop_min(node.right)
                m.left, m.right = node.left, node.right
                node = m
            return self.rebalance(node)

        self.root = rec(self.root)
        if removed[0]:
            self.n -= 1
        return removed[0]

    def search(self, key):
        cur = self.root
        while cur:
            self.c.comparisons += 1
            if key == cur.key:
                return True
            cur = cur.left if key < cur.key else cur.right
        return False


def min_avl_nodes(h):
    """N(0) = 0, N(1) = 1, N(h) = N(h-1) + N(h-2) + 1. 높이 h 인 AVL 트리가 가질 수 있는 가장 적은 노드 수."""
    a, b = 0, 1
    if h == 0:
        return 0
    for _ in range(h - 1):
        a, b = b, a + b + 1
    return b


def fibonacci_tree_keys(h):
    """높이 h 이면서 노드가 N(h)개뿐인 AVL 트리(피보나치 트리)를 만들고, 그 트리를 BST 에 넣었을 때
    같은 모양이 되는 넣는 순서(너비 우선)를 돌려준다. 키는 1..N(h)."""
    def build(h, lo):
        if h == 0:
            return None, lo
        left, nxt = build(h - 1, lo)  # 왼쪽이 더 높은 쪽
        node = Node(nxt)
        node.left = left
        right, nxt2 = build(h - 2, nxt + 1) if h >= 2 else (None, nxt + 1)
        node.right = right
        _fix(node)
        return node, nxt2
    root, _ = build(h, 1)
    order, q = [], [root] if root else []
    while q:
        nd = q.pop(0)
        order.append(nd.key)
        q += [x for x in (nd.left, nd.right) if x]
    return order, root


# ---------------------------------------------------------------- 이진 힙 (최소 힙)

class BinaryHeap:
    """배열 a 에서 a[(i-1)//2] <= a[i] 를 지킨다. 비교 한 번마다 counter.sift += 1."""

    def __init__(self, counter=None):
        self.a = []
        self.c = _c(counter)

    def _less(self, x, y):
        self.c.sift += 1
        return x < y

    def push(self, x):
        a = self.a
        a.append(x)
        i = len(a) - 1
        while i > 0:
            p = (i - 1) // 2
            if self._less(a[i], a[p]):
                a[i], a[p] = a[p], a[i]
                i = p
            else:
                break

    def _sift_down(self, i):
        a, n = self.a, len(self.a)
        while True:
            l, r, s = 2 * i + 1, 2 * i + 2, i
            if l < n and self._less(a[l], a[s]):
                s = l
            if r < n and self._less(a[r], a[s]):
                s = r
            if s == i:
                return
            a[i], a[s] = a[s], a[i]
            i = s

    def pop(self):
        a = self.a
        top = a[0]
        last = a.pop()
        if a:
            a[0] = last
            self._sift_down(0)
        return top

    def heapify(self, xs):
        """Floyd 식: 마지막 부모부터 뿌리까지 거꾸로 sift-down. 비교 수는 2n 이하."""
        self.a = list(xs)
        for i in range(len(self.a) // 2 - 1, -1, -1):
            self._sift_down(i)

    def __len__(self):
        return len(self.a)


def check_heap(a):
    bad = [i for i in range(1, len(a)) if a[(i - 1) // 2] > a[i]]
    return ("부모 <= 자식", not bad, f"깨진 자리 {bad[:5]}" if bad else "")


# ---------------------------------------------------------------- 넣는 순서

def orders(name, n, rng=None):
    keys = list(range(1, n + 1))
    if name == "sorted":
        return keys
    if name == "zigzag":  # 1, n, 2, n-1, ...  양쪽 끝에서 번갈아 넣는다
        out, lo, hi = [], 1, n
        while lo <= hi:
            out.append(lo)
            if lo != hi:
                out.append(hi)
            lo, hi = lo + 1, hi - 1
        return out
    if name == "random":
        rng = rng or random.Random(0)
        rng.shuffle(keys)
        return keys
    raise ValueError(name)


def build(kind, keys, counter=None):
    c = _c(counter)
    t = BST(c) if kind == "bst" else AVL(c)
    for k in keys:
        t.insert(k)
    return t, c


def mean_search_comparisons(root, keys):
    """keys 를 모두 찾을 때 비교 수 평균 = 깊이의 평균."""
    d = depths(root)
    return sum(d[k] for k in keys) / len(keys)


# ---------------------------------------------------------------- 위젯 트레이스

def snapshot(root):
    """[[키, 중위 순위, 깊이, 균형 인수], ...] 와 간선 [[부모 키, 자식 키], ...]."""
    if root is None:
        return [], []
    rank = {k: i for i, k in enumerate(inorder(root))}
    d = depths(root)
    bfs = balance_factors(root)
    nodes = [[k, rank[k], d[k], bfs[k]] for k in sorted(d)]
    edges = []
    for nd in _nodes(root):
        for ch in (nd.left, nd.right):
            if ch:
                edges.append([nd.key, ch.key])
    edges.sort()
    return nodes, edges


def trace_inserts(keys, kind):
    """kind = 'bst' | 'avl'. 단계: [설명, 넣은 키, 노드, 간선, 높이, 누적 비교, 누적 회전, 깨진 노드 목록, 중위 정렬 여부, 단계 종류]."""
    c = Counter()
    steps = [["빈 트리", None, [], [], 0, 0, 0, [], True, "start"]]
    holder = {}

    def hook(ev, node):
        if ev == "violation":
            nodes, edges = snapshot(holder["t"].root_for_snapshot())
            bad = [k for k, _, _, b in nodes if abs(b) > 1]
            steps.append([f"키 {holder['k']} 넣은 직후: 노드 {node.key}에서 균형 인수 "
                          f"{dict((n[0], n[3]) for n in nodes)[node.key]:+d}", holder["k"], nodes, edges,
                          max([n[2] for n in nodes] or [0]), c.comparisons, c.rotations, bad, True, "violation"])
        else:
            holder["rot"] = ev.split(":", 1)[1]

    class _AVL(AVL):
        def root_for_snapshot(self):
            return self.root

    t = BST(c) if kind == "bst" else _AVL(c, hook=hook)
    holder["t"] = t
    for k in keys:
        holder["k"], holder["rot"] = k, None
        before = c.comparisons
        # AVL: 회전 직전 스냅숏(hook)은 아직 바뀌지 않은 root 에서 찍힌다. 새 리프는 재귀가 돌아오기 전에
        # 이미 붙어 있고, 넣기 한 번에 회전은 많아야 한 곳에서만 일어나므로 그 순간의 구조는 온전하다.
        t.insert(k)
        nodes, edges = snapshot(t.root)
        bad = [kk for kk, _, _, b in nodes if abs(b) > 1]
        desc = f"키 {k} 넣기: 비교 {c.comparisons - before}번"
        typ = "insert"
        if holder["rot"]:
            desc = f"{holder['rot']}으로 복구(키 {k} 넣기 비교 {c.comparisons - before}번)"
            typ = "rotated"
        steps.append([desc, k, nodes, edges, height(t.root), c.comparisons, c.rotations, bad,
                      check_bst_order(t.root)[1], typ])
    return {"kind": kind, "keys": list(keys), "comparisons": c.comparisons, "rotations": c.rotations,
            "height": height(t.root), "steps": steps}
