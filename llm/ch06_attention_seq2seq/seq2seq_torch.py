"""6장 PyTorch 대응 구현.

seq2seq_np.py와 같은 식을 배치 텐서로 옮기고, 학습 비교에 쓸 인코더-디코더 하나를 정의한다.
- 점수 함수 셋(additive / dot / general)은 NumPy 구현과 한 줄씩 대응한다.
- Seq2Seq(score="fixed")는 어텐션 없이 인코더 마지막 상태 h_L 하나만 문맥으로 쓰는 기준선이다.
- 학습은 정답 토큰을 디코더 입력으로 넣는(teacher forcing) 한 번의 nn.GRU 호출로 하고,
  평가는 step()을 한 걸음씩 부르는 탐욕 디코딩으로 한다. 두 경로가 같은 값을 내는지 테스트한다.
"""

import torch
import torch.nn as nn


def additive_score(S, H, Ws, Wh, v):
    """S: (B, T, d), H: (B, L, d), Ws, Wh: (d_a, d), v: (d_a,). 반환 (B, T, L)."""
    return torch.tanh((S @ Ws.T)[:, :, None, :] + (H @ Wh.T)[:, None, :, :]) @ v


def dot_score(S, H):
    return S @ H.transpose(-2, -1)


def general_score(S, H, W):
    return (S @ W) @ H.transpose(-2, -1)


def attend(scores, H):
    """scores: (B, T, L) -> alpha (B, T, L), c (B, T, d)."""
    alpha = torch.softmax(scores, dim=-1)
    return alpha @ H, alpha


class Seq2Seq(nn.Module):
    def __init__(self, vocab, d=64, e=32, score="dot"):
        super().__init__()
        self.score = score
        self.emb = nn.Embedding(vocab + 1, e)  # 마지막 번호는 디코더 시작 토큰 BOS
        self.bos = vocab
        self.enc = nn.GRU(e, d, batch_first=True)
        self.dec = nn.GRU(e, d, batch_first=True)
        if score == "additive":
            self.Ws = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.Wh = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.v = nn.Parameter(torch.randn(d) / d**0.5)
        elif score == "general":
            self.W = nn.Parameter(torch.randn(d, d) / d**0.5)
        self.Wc = nn.Linear(2 * d, d, bias=False)
        self.Wo = nn.Linear(d, vocab, bias=False)

    def score_params(self):
        return {"additive": lambda: (self.Ws, self.Wh, self.v), "general": lambda: (self.W,)}.get(self.score, lambda: ())()

    def encode_embedded(self, xe):
        H, hL = self.enc(xe)
        return H, hL

    def _context(self, S, H):
        if self.score == "fixed":
            return H[:, -1:, :].expand(-1, S.shape[1], -1), None
        fn = {"additive": additive_score, "dot": dot_score, "general": general_score}[self.score]
        return attend(fn(S, H, *self.score_params()), H)

    def decode_teacher_forced(self, H, hL, tgt_in):
        S, _ = self.dec(self.emb(tgt_in), hL)  # 디코더 상태 s_1..s_T를 한 번에
        C, alpha = self._context(S, H)
        return self.Wo(torch.tanh(self.Wc(torch.cat([C, S], dim=-1)))), alpha

    def forward(self, src, tgt_in):
        H, hL = self.encode_embedded(self.emb(src))
        return self.decode_teacher_forced(H, hL, tgt_in)

    def step(self, y_prev, s_prev, H):
        """디코더 한 걸음. y_prev: (B,), s_prev: (1, B, d). 반환 logits (B, V), s (1, B, d), alpha (B, L)."""
        S, s = self.dec(self.emb(y_prev)[:, None, :], s_prev)
        C, alpha = self._context(S, H)
        logits = self.Wo(torch.tanh(self.Wc(torch.cat([C, S], dim=-1))))[:, 0]
        return logits, s, (None if alpha is None else alpha[:, 0])

    @torch.no_grad()
    def greedy(self, src, T):
        H, s = self.encode_embedded(self.emb(src))
        y = torch.full((src.shape[0],), self.bos, dtype=torch.long)
        preds, alphas = [], []
        for _ in range(T):
            logits, s, alpha = self.step(y, s, H)
            y = logits.argmax(-1)
            preds.append(y)
            alphas.append(alpha)
        A = None if alphas[0] is None else torch.stack(alphas, 1)
        return torch.stack(preds, 1), A
