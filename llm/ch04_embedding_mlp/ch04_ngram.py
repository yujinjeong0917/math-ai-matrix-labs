"""보간 트라이그램 기준선. 2장 설계도(docs/tracks/llm.md)의 보간 n-gram 식을 이 장에 필요한 만큼만 직접 구현했다(2장 코드와 공유하지 않는다).

P(w | u, v) = l3 * P_ML(w | u, v) + l2 * P_ML(w | v) + l1 * P_add1(w)

문맥마다 따로 센 상대빈도를 섞을 뿐이라, 학습 데이터에 없는 (v, w)는 l1 * P(w)로만 확률을 얻는다.
이 덧셈이 "백오프로만 확률을 얻는다"는 말의 실제 모양이다.
"""

from collections import Counter
from itertools import product

import numpy as np


class InterpolatedTrigram:
    def __init__(self, V):
        self.V = V
        # 초기 가중치 (l3, l2, l1)는 균등. tune()이 검증 세트로 다시 고른다.
        self.lambdas = (1 / 3, 1 / 3, 1 / 3)

    def fit(self, ctx, tgt):
        # 카운트: c3 = (u, v, w), c2ctx = (u, v) 문맥, c2 = (v, w), c1ctx = v 문맥, c1 = w.
        self.c3 = Counter(zip(ctx[:, 0].tolist(), ctx[:, 1].tolist(), tgt.tolist()))
        self.c2ctx = Counter(zip(ctx[:, 0].tolist(), ctx[:, 1].tolist()))
        self.c2 = Counter(zip(ctx[:, 1].tolist(), tgt.tolist()))
        self.c1ctx = Counter(ctx[:, 1].tolist())
        self.c1 = Counter(tgt.tolist())
        self.N = len(tgt)
        return self

    def prob(self, u, v, w):
        l3, l2, l1 = self.lambdas
        # 문맥을 본 적이 없으면 분모가 0이라 해당 항을 0으로 둔다.
        p3 = self.c3[(u, v, w)] / self.c2ctx[(u, v)] if self.c2ctx[(u, v)] else 0.0
        p2 = self.c2[(v, w)] / self.c1ctx[v] if self.c1ctx[v] else 0.0
        # 식: P_add1(w) = (c(w) + 1) / (N + |V|). 항상 > 0이라 보간 확률이 0이 되지 않는다.
        p1 = (self.c1[w] + 1) / (self.N + self.V)  # 덧셈 평활: 0이 되지 않게
        # 식 P = l3 P3 + l2 P2 + l1 P1. 세 가중치의 합이 1이면 분포 합이 1.
        return l3 * p3 + l2 * p2 + l1 * p1

    def probs(self, ctx, tgt):
        return np.array([self.prob(u, v, w) for (u, v), w in zip(ctx.tolist(), tgt.tolist())])

    def nll(self, ctx, tgt):
        return float(-np.mean(np.log(self.probs(ctx, tgt))))

    def tune(self, ctx, tgt, step=0.05, floor=0.05):
        """검증 세트 NLL이 가장 낮은 (l3, l2, l1)을 격자에서 고른다. 각 값은 floor 이상."""
        best = None
        # l3, l2를 격자로 돌고 l1 = 1 - l3 - l2. 부동소수점 누적 오차를 피하려고 round한다.
        grid = np.round(np.arange(floor, 1.0 + 1e-9, step), 6)
        for l3, l2 in product(grid, grid):
            l1 = round(1.0 - l3 - l2, 6)
            # l1이 floor 미만이면 건너뛴다(1e-9는 부동소수점 허용 오차).
            if l1 < floor - 1e-9:
                continue
            self.lambdas = (l3, l2, l1)
            loss = self.nll(ctx, tgt)
            if best is None or loss < best[0]:
                best = (loss, self.lambdas)
        self.lambdas = best[1]
        return self.lambdas

    def stored_entries(self):
        """실제로 저장된 (0이 아닌) 카운트 칸 수."""
        return len(self.c3) + len(self.c2) + len(self.c1)

    def table_size(self, n=3):
        """문맥마다 독립 확률표를 둘 때의 칸 수: |V|^n + |V|^(n-1) + |V|."""
        return sum(self.V**k for k in range(1, n + 1))
