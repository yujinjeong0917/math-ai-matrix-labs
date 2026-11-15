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
    # (B, T, 1, d_a) + (B, 1, L, d_a) -> (B, T, L, d_a)로 브로드캐스트 -> tanh -> v 내적 -> (B, T, L). NumPy 버전의 한 줄에 대응.
    return torch.tanh((S @ Ws.T)[:, :, None, :] + (H @ Wh.T)[:, None, :, :]) @ v


def dot_score(S, H):
    """S: (B, T, d), H: (B, L, d) -> (B, T, L). 모든 디코더 위치와 인코더 위치 쌍의 내적."""
    return S @ H.transpose(-2, -1)


def general_score(S, H, W):
    """(S W) H^T. W: (d, d). 반환 (B, T, L)."""
    return (S @ W) @ H.transpose(-2, -1)


def attend(scores, H):
    """scores: (B, T, L) -> alpha (B, T, L), c (B, T, d)."""
    alpha = torch.softmax(scores, dim=-1)
    return alpha @ H, alpha


class Seq2Seq(nn.Module):
    """GRU 인코더-디코더 + 선택형 어텐션(fixed/additive/dot/general). 문맥과 디코더 상태를 이어 출력층에 넣는다."""
    def __init__(self, vocab, d=64, e=32, score="dot"):
        super().__init__()
        self.score = score
        # 번호 vocab을 추가해 BOS로 쓴다. 인코더 입력 토큰은 0..vocab-1.
        self.emb = nn.Embedding(vocab + 1, e)  # 마지막 번호는 디코더 시작 토큰 BOS
        self.bos = vocab
        # 인코더와 디코더는 가중치를 공유하지 않는 별도 GRU. 입력 (B, n, e), 은닉 d.
        self.enc = nn.GRU(e, d, batch_first=True)
        self.dec = nn.GRU(e, d, batch_first=True)
        if score == "additive":
            # 1/sqrt(d) 스케일 정규분포 초기화(모형의 선택). d_a = d로 둔다.
            self.Ws = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.Wh = nn.Parameter(torch.randn(d, d) / d**0.5)
            self.v = nn.Parameter(torch.randn(d) / d**0.5)
        elif score == "general":
            self.W = nn.Parameter(torch.randn(d, d) / d**0.5)
        # [c; s] (2d) -> d.
        self.Wc = nn.Linear(2 * d, d, bias=False)
        # 출력 로짓은 vocab개(BOS는 출력하지 않는다).
        self.Wo = nn.Linear(d, vocab, bias=False)

    def score_params(self):
        """현재 점수 함수가 쓰는 가중치 튜플(dot과 fixed는 빈 튜플)."""
        return {"additive": lambda: (self.Ws, self.Wh, self.v), "general": lambda: (self.W,)}.get(self.score, lambda: ())()

    def encode_embedded(self, xe):
        """임베딩된 입력을 GRU로 읽어 H (B, L, d)와 마지막 은닉 hL (1, B, d)를 돌려준다."""
        H, hL = self.enc(xe)
        return H, hL

    def _context(self, S, H):
        """S: (B, T, d) 디코더 상태로 문맥 C (B, T, d)와 alpha (B, T, L)를 만든다. fixed면 마지막 인코더 상태를 T번 복제하고 alpha는 None."""
        if self.score == "fixed":
            # (B, 1, d)를 (B, T, d)로 복제해 모든 디코더 위치가 같은 문맥을 본다(어텐션 없음).
            return H[:, -1:, :].expand(-1, S.shape[1], -1), None
        fn = {"additive": additive_score, "dot": dot_score, "general": general_score}[self.score]
        return attend(fn(S, H, *self.score_params()), H)

    def decode_teacher_forced(self, H, hL, tgt_in):
        """정답 토큰을 입력으로 넣는 한 번의 GRU 호출. 반환 (logits (B, T, V), alpha)."""
        # 디코더 초기 상태 = 인코더 마지막 상태 hL. S: (B, T, d).
        S, _ = self.dec(self.emb(tgt_in), hL)  # 디코더 상태 s_1..s_T를 한 번에
        C, alpha = self._context(S, H)
        return self.Wo(torch.tanh(self.Wc(torch.cat([C, S], dim=-1)))), alpha

    def forward(self, src, tgt_in):
        """학습용 순전파. src: (B, L), tgt_in: (B, T) -> (logits, alpha)."""
        H, hL = self.encode_embedded(self.emb(src))
        return self.decode_teacher_forced(H, hL, tgt_in)

    def step(self, y_prev, s_prev, H):
        """디코더 한 걸음. y_prev: (B,), s_prev: (1, B, d). 반환 logits (B, V), s (1, B, d), alpha (B, L)."""
        # 한 걸음이라 길이 1 시퀀스로 넣는다: (B, 1, e). s는 다음 걸음의 s_prev.
        S, s = self.dec(self.emb(y_prev)[:, None, :], s_prev)
        C, alpha = self._context(S, H)
        logits = self.Wo(torch.tanh(self.Wc(torch.cat([C, S], dim=-1))))[:, 0]
        return logits, s, (None if alpha is None else alpha[:, 0])

    @torch.no_grad()
    def greedy(self, src, T):
        """탐욕 디코딩: BOS에서 시작해 매 걸음 argmax를 다음 입력으로 쓴다(오차가 누적될 수 있다). 반환 (예측 (B, T), alpha (B, T, L) 또는 None)."""
        H, s = self.encode_embedded(self.emb(src))
        # 첫 입력은 BOS 토큰.
        y = torch.full((src.shape[0],), self.bos, dtype=torch.long)
        preds, alphas = [], []
        for _ in range(T):
            logits, s, alpha = self.step(y, s, H)
            y = logits.argmax(-1)
            preds.append(y)
            alphas.append(alpha)
        A = None if alphas[0] is None else torch.stack(alphas, 1)
        return torch.stack(preds, 1), A
