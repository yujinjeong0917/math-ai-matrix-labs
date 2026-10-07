"""토큰 단위가 다른 모델을 같은 잣대로 재는 도구: 보간 trigram + 글자당 비트(bpc).

2장의 보간 n-gram을 이 장에서 쓸 만큼만 다시 쓴 것이다(2장과 동시에 작성되어 import하지 않음).
재귀 보간(Jelinek-Mercer 꼴):
  P1(w)       = l1 * c(w)/N + (1 - l1) / |V|
  P2(w|v)     = l2 * c(v,w)/c(v) + (1 - l2) * P1(w)       (c(v) = 0이면 P1)
  P3(w|u,v)   = l3 * c(u,v,w)/c(u,v) + (1 - l3) * P2(w|v)   (c(u,v) = 0이면 P2)
어느 단계든 어휘 전체에 대한 합이 1인 분포다.

글자당 비트 = (모든 토큰의 -log2 P 합 + <unk>의 철자 비용) / 원문 글자 수.
<unk> 하나가 가린 글자 k개에는 k * log2(U) 비트를 더 매긴다. U는 글자 우주의 크기
(en_like 26, ko_like 한글 음절 11,172)이고, '가린 글자를 아무 정보 없이 하나씩 받아 적는 비용'이다.
"""

from collections import Counter

import numpy as np

BOS, EOS = "<s>", "</s>"


class InterpTrigram:
    def __init__(self, train_seqs, vocab):
        self.vocab = set(vocab) | {EOS}
        self.V = len(self.vocab)
        self.c1, self.c2, self.c3 = Counter(), Counter(), Counter()
        self.h1, self.h2 = Counter(), Counter()  # 문맥 카운트
        for seq in train_seqs:
            s = [BOS, BOS] + list(seq) + [EOS]
            for i in range(2, len(s)):
                u, v, w = s[i - 2], s[i - 1], s[i]
                self.c1[w] += 1
                self.c2[v, w] += 1
                self.c3[u, v, w] += 1
                self.h1[v] += 1
                self.h2[u, v] += 1
        self.N = sum(self.c1.values())

    def components(self, seqs):
        """토큰마다 (ML3, 3문맥 있음, ML2, 2문맥 있음, ML1)을 배열로. 람다 격자 탐색을 벡터로 하려고."""
        rows = []
        for seq in seqs:
            s = [BOS, BOS] + list(seq) + [EOS]
            for i in range(2, len(s)):
                u, v, w = s[i - 2], s[i - 1], s[i]
                h2, h1 = self.h2[u, v], self.h1[v]
                rows.append((self.c3[u, v, w] / h2 if h2 else 0.0, h2 > 0,
                             self.c2[v, w] / h1 if h1 else 0.0, h1 > 0,
                             self.c1[w] / self.N))
        a = np.array(rows, dtype=float)
        return a[:, 0], a[:, 1].astype(bool), a[:, 2], a[:, 3].astype(bool), a[:, 4]

    def probs(self, comps, lams):
        m3, h3, m2, h2, m1 = comps
        l1, l2, l3 = lams
        p1 = l1 * m1 + (1 - l1) / self.V
        p2 = np.where(h2, l2 * m2 + (1 - l2) * p1, p1)
        return np.where(h3, l3 * m3 + (1 - l3) * p2, p2)

    def prob(self, u, v, w, lams):
        """한 칸 확률(테스트에서 분포 합 1 확인용)."""
        l1, l2, l3 = lams
        p1 = l1 * self.c1[w] / self.N + (1 - l1) / self.V
        p2 = l2 * self.c2[v, w] / self.h1[v] + (1 - l2) * p1 if self.h1[v] else p1
        return l3 * self.c3[u, v, w] / self.h2[u, v] + (1 - l3) * p2 if self.h2[u, v] else p2


L23 = (0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999)
L1 = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999)
LAMBDA_GRID = [(l1, l2, l3) for l1 in L1 for l2 in L23 for l3 in L23]


def tune_lambdas(model, dev_seqs):
    """검증 세트 로그가능도가 가장 큰 (l1, l2, l3). 격자 가장자리에 걸렸는지도 돌려준다.

    l2 = 0 또는 l3 = 0은 정당한 가장자리다(그 차수의 카운트가 검증 세트에서 도움이 안 됨).
    """
    comps = model.components(dev_seqs)
    best = min(LAMBDA_GRID, key=lambda lam: -np.log2(model.probs(comps, lam)).sum())
    edges = {"l1": best[0] in (L1[0], L1[-1]), "l2": best[1] in (L23[0], L23[-1]),
             "l3": best[2] in (L23[0], L23[-1])}
    return best, edges


def bits_per_char(model, seqs, lams, n_chars, unk_chars, U):
    """seqs의 전체 비트 / 원문 글자 수. unk_chars: <unk>가 가린 글자 수 합."""
    p = model.probs(model.components(seqs), lams)
    n_tok = len(p)
    bits_model = float(-np.log2(p).sum())
    return {
        "bpc": (bits_model + unk_chars * np.log2(U)) / n_chars,
        "bpc_unk_free": bits_model / n_chars,  # 철자 비용을 빼면 <unk>가 공짜가 된다(착시)
        "token_ppl": float(2 ** (bits_model / n_tok)),
        "tokens": n_tok,
    }
