"""1장 PyTorch 대응 구현.

같은 확률표를 셈 대신 학습으로 얻는다. nn.Embedding(V, V)의 a번째 행을
"앞 단어가 a일 때 다음 단어 점수(로짓)"로 두고, softmax로 확률을 만든 뒤
교차엔트로피를 줄인다. 교차엔트로피의 최솟값은 정확히 c(a,b)/c(a)에서 나오므로
(MLE와 같은 문제), 수렴한 softmax 표가 bigram_np.bigram_mle 결과와 같아야 한다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLM(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.logits = nn.Embedding(V, V)
        nn.init.zeros_(self.logits.weight)  # 시작은 모든 다음 단어가 같은 확률

    def forward(self, prev):
        return self.logits(prev)

    def table(self):
        return torch.softmax(self.logits.weight, dim=-1)


def fit_bigram(counts, steps=3000, lr=0.1, seed=0):
    """counts: (V, V) 카운트 표. 모든 (앞, 다음) 쌍을 한 배치로 넣은 전체 배치 학습.

    토큰 하나하나를 넣는 대신 쌍마다 등장 횟수를 가중치로 곱한다. 손실은 같다:
    L = -(1/N) sum_{a,b} c(a,b) log softmax(z_a)_b
    """
    torch.manual_seed(seed)
    c = torch.as_tensor(counts, dtype=torch.float64)
    V = c.shape[0]
    model = BigramLM(V).double()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    prev = torch.arange(V)
    N = c.sum()
    losses = []
    for _ in range(steps):
        logp = F.log_softmax(model(prev), dim=-1)
        loss = -(c * logp).sum() / N
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return model, losses


def perplexity_torch(P, sentences_ids):
    """같은 PPL을 F.nll_loss로 다시 계산한다(NumPy 구현과 대조용)."""
    P = torch.as_tensor(P, dtype=torch.float64)
    prev = torch.cat([torch.as_tensor(s[:-1]) for s in sentences_ids])
    nxt = torch.cat([torch.as_tensor(s[1:]) for s in sentences_ids])
    with torch.no_grad():
        nll = F.nll_loss(torch.log(P[prev]), nxt)
    return float(torch.exp(nll))
