"""강화학습 6장 PyTorch 대응 구현.

1) 전이 기록 재생: control()이 남긴 기록 (episode, t, s, a, r, s_next, done, fell)을 텐서로 받아
   Sarsa / Q-러닝 / Expected Sarsa 갱신을 같은 순서로 다시 한다. 결과 Q가 NumPy와 atol 1e-12 안이어야 한다.
2) 무작위 보행 TD(0)·상수 alpha MC를 시드 B개에 대해 한꺼번에: 판을 [B, T] 텐서로 채워 시간 방향으로만 반복한다.
float64로 계산한다.
"""

import torch

DT = torch.float64


def log_to_tensors(log):
    cols = list(zip(*log))
    s = torch.tensor(cols[2], dtype=torch.long)
    a = torch.tensor(cols[3], dtype=torch.long)
    r = torch.tensor(cols[4], dtype=DT)
    s2 = torch.tensor(cols[5], dtype=torch.long)
    done = torch.tensor(cols[6], dtype=torch.bool)
    return s, a, r, s2, done


def eps_greedy_probs_t(q_row, eps):
    """동률인 최댓값 행동들이 (1 - eps)를 나눠 갖는 epsilon-탐욕 확률(NumPy eps_greedy_probs와 같은 규칙)."""
    A = q_row.shape[0]
    ties = (q_row == q_row.max()).to(DT)
    return torch.full((A,), eps / A, dtype=DT) + ties * ((1.0 - eps) / ties.sum())


def replay_t(log, S, A, method, alpha, gamma, eps=0.1, q0=0.0):
    s, a, r, s2, done = log_to_tensors(log)
    Q = torch.full((S, A), float(q0), dtype=DT)
    n = s.shape[0]
    for i in range(n):
        si, ai, s2i = s[i], a[i], s2[i]
        if done[i]:
            target = r[i]
        elif method == "sarsa":
            target = r[i] + gamma * Q[s2i, a[i + 1]]
        elif method == "q":
            target = r[i] + gamma * Q[s2i].max()
        elif method == "expected":
            p = eps_greedy_probs_t(Q[s2i], eps)
            target = r[i] + gamma * (p * Q[s2i]).sum()
        else:
            raise ValueError(method)
        Q[si, ai] = Q[si, ai] + alpha * (target - Q[si, ai])
    return Q


def pad_walks(batch_of_episodes):
    """시드마다 판 하나: [(s, r, s2, done), ...]의 목록 B개를 [B, T] 텐서로. 짧은 판의 뒤는 mask=False."""
    B = len(batch_of_episodes)
    T = max(len(ep) for ep in batch_of_episodes)
    S = torch.zeros((B, T), dtype=torch.long)
    S2 = torch.zeros((B, T), dtype=torch.long)
    R = torch.zeros((B, T), dtype=DT)
    D = torch.zeros((B, T), dtype=torch.bool)
    M = torch.zeros((B, T), dtype=torch.bool)
    for b, ep in enumerate(batch_of_episodes):
        for t, (s, r, s2, d) in enumerate(ep):
            S[b, t], R[b, t], S2[b, t], D[b, t], M[b, t] = s, r, s2, d, True
    return S, R, S2, D, M


def td0_batch(runs, n_states, alpha, gamma, v0=0.0):
    """runs[b] = 시드 b의 판 목록. 판 번호 k마다 B개 시드의 k번째 판을 한꺼번에 넣는다.

    시드마다 V가 따로 있다(V: [B, n_states]). 한 시드 안의 갱신 순서는 NumPy td0과 같다.
    반환: 판마다 V의 사본 [K, B, n_states].
    """
    B, K = len(runs), len(runs[0])
    V = torch.full((B, n_states), float(v0), dtype=DT)
    rows = torch.arange(B)
    hist = []
    for k in range(K):
        S, R, S2, D, M = pad_walks([runs[b][k] for b in range(B)])
        for t in range(S.shape[1]):
            s, s2 = S[:, t], S2[:, t]
            nxt = torch.where(D[:, t], torch.zeros(B, dtype=DT), V[rows, s2])
            target = R[:, t] + gamma * nxt
            new = V[rows, s] + alpha * (target - V[rows, s])
            V[rows, s] = torch.where(M[:, t], new, V[rows, s])
        hist.append(V.clone())
    return torch.stack(hist)


def mc_batch(runs, n_states, alpha, gamma, v0=0.0):
    """상수 alpha 모든 방문 MC를 같은 방식으로. 리턴은 끝에서부터 G_t = r_t + gamma G_{t+1}."""
    B, K = len(runs), len(runs[0])
    V = torch.full((B, n_states), float(v0), dtype=DT)
    rows = torch.arange(B)
    hist = []
    for k in range(K):
        S, R, S2, D, M = pad_walks([runs[b][k] for b in range(B)])
        T = S.shape[1]
        G = torch.zeros((B, T), dtype=DT)
        g = torch.zeros(B, dtype=DT)
        for t in range(T - 1, -1, -1):
            g = torch.where(M[:, t], R[:, t] + gamma * g, g)
            G[:, t] = g
        for t in range(T):
            s = S[:, t]
            new = V[rows, s] + alpha * (G[:, t] - V[rows, s])
            V[rows, s] = torch.where(M[:, t], new, V[rows, s])
        hist.append(V.clone())
    return torch.stack(hist)
