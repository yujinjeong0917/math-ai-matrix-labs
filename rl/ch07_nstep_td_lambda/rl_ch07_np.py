"""강화학습 7장 NumPy 최소 구현: n스텝 TD, λ-리턴, 적격 흔적 TD(λ).

표 기반 상태가치 예측만 다룬다. 전이 하나는 (s, r, s_next, done).
- n_step_td: 실제 보상 n개 + n걸음 뒤 상태의 지금 추정값을 목표로 쓴다(n=None이면 끝까지 = 상수 alpha MC).
- lambda_returns: 한 판의 λ-리턴 G^λ_t를 뒤에서부터 계산한다(가치 V는 고정).
- td_lambda: 걸음마다 흔적을 γλ배로 줄이고 방문한 칸에 1을 더하거나(accumulating) 1로 덮어써서(replacing)
  지금의 TD 오차를 지나온 칸들에 나눠 준다. online=False면 판이 끝날 때 한꺼번에 더한다.
- lambda_return_summed / offline_lambda_return: 전방 관점 두 가지(판마다 합산 / 판 끝에 차례로).
- *_batch: 같은 계산을 (시드 × 보폭) 행으로 한꺼번에 돌리는 실험용 판. 테스트에서 위의 기준 구현과 대조한다.
"""

import numpy as np


# ---------------- 기준 구현 (한 시드, 읽기 쉬운 판) ----------------

def n_step_return(ep, V, tau, n, gamma):
    """G_{tau:tau+n}. ep[t] = (S_t, R_{t+1}, S_{t+1}, done). n=None이면 판 끝까지."""
    T = len(ep)
    end = T if n is None else min(tau + n, T)
    G = 0.0
    for i in range(tau, end):                 # 실제 보상 R_{tau+1} ... R_end
        G += gamma ** (i - tau) * ep[i][1]
    if n is not None and tau + n < T:         # 판이 안 끝났으면 n걸음 뒤 상태의 지금 값으로 메운다
        G += gamma ** n * V[ep[tau + n][0]]
    return G


def n_step_td(episodes, n_states, n, alpha, gamma=1.0, v0=0.0, record=None):
    """n스텝 TD. 걸음 tau+n-1에 S_tau를 고치는 온라인 알고리즘과 같은 순서로 고친다.

    온라인 판에서도 V가 바뀌는 건 이 갱신뿐이고 tau 순서대로 일어나므로, tau를 차례로 도는 것과 결과가 같다.
    record가 리스트면 판이 끝날 때마다 V 사본을 넣는다.
    """
    V = [v0] * n_states
    for ep in episodes:
        for tau in range(len(ep)):
            s = ep[tau][0]
            V[s] += alpha * (n_step_return(ep, V, tau, n, gamma) - V[s])
        if record is not None:
            record.append(np.array(V))
    return np.array(V)


def td0(episodes, n_states, alpha, gamma=1.0, v0=0.0):
    """6장 rl_ch06_td_np.td0에서 복사. 테스트에서 n=1, λ=0과 비교한다."""
    V = [v0] * n_states
    for ep in episodes:
        for s, r, s2, done in ep:
            target = r if done else r + gamma * V[s2]
            V[s] += alpha * (target - V[s])
    return np.array(V)


def mc_constant_alpha_every_visit(episodes, n_states, alpha, gamma=1.0, v0=0.0):
    """판이 끝난 뒤 앞에서부터 모든 방문에 대해 V(S_t) += alpha (G_t - V(S_t)). 6장 mc_constant_alpha와 같은 순서."""
    V = [v0] * n_states
    for ep in episodes:
        G, Gs = 0.0, []
        for s, r, s2, done in reversed(ep):
            G = r + gamma * G
            Gs.append(G)
        Gs.reverse()
        for (s, *_), G in zip(ep, Gs):
            V[s] += alpha * (G - V[s])
    return np.array(V)


def lambda_returns(ep, V, lam, gamma):
    """한 판의 G^λ_t (t = 0..T-1). V는 고정.

    G^λ_t = R_{t+1} + γ[(1-λ) V(S_{t+1}) + λ G^λ_{t+1}], 종료 직전 걸음은 G^λ_{T-1} = R_T.
    (1-λ)Σ λ^{n-1} G_{t:t+n} 을 직접 더한 것과 같다(테스트에서 대조).
    """
    T = len(ep)
    G = np.zeros(T)
    nxt = 0.0
    for t in range(T - 1, -1, -1):
        s, r, s2, done = ep[t]
        if done:
            G[t] = r
        else:
            G[t] = r + gamma * ((1.0 - lam) * V[s2] + lam * nxt)
        nxt = G[t]
    return G


def lambda_return_by_definition(ep, V, lam, gamma):
    """전방 관점 정의 그대로: (1-λ) Σ_{n=1}^{T-t-1} λ^{n-1} G_{t:t+n} + λ^{T-t-1} G_t."""
    T = len(ep)
    out = np.zeros(T)
    for t in range(T):
        acc = 0.0
        for n in range(1, T - t):
            acc += (1.0 - lam) * lam ** (n - 1) * n_step_return(ep, V, t, n, gamma)
        acc += lam ** (T - t - 1) * n_step_return(ep, V, t, None, gamma)
        out[t] = acc
    return out


def lambda_return_summed(episodes, n_states, lam, alpha, gamma=1.0, v0=0.0, record=None):
    """판 동안 V를 고정해 G^λ_t를 모두 구하고, 판이 끝나면 모든 걸음의 고칠 양을 더해 한 번에 넣는다.

    Sutton(1988) 식 (1)처럼 판(sequence)마다 Δw를 모아 더하는 방식이다. 같은 칸을 k번 지나면
    그 칸은 한 판에 α를 k번 더한 만큼 움직인다. 누적 흔적 TD(λ)의 오프라인 판과 정확히 같다(테스트).
    """
    V = np.full(n_states, v0, dtype=float)
    for ep in episodes:
        G = lambda_returns(ep, V, lam, gamma)
        dV = np.zeros(n_states)
        for (s, *_), g in zip(ep, G):
            dV[s] += alpha * (g - V[s])
        V += dV
        if record is not None:
            record.append(V.copy())
    return V


def offline_lambda_return(episodes, n_states, lam, alpha, gamma=1.0, v0=0.0, record=None):
    """오프라인 λ-리턴 알고리즘(Sutton·Barto 2판 식 12.4의 표 기반 판).

    판 동안은 고치지 않고 G^λ_t를 판 시작의 V로 구한다. 판이 끝나면 t = 0, 1, ... 차례로
    V(S_t) <- V(S_t) + α (G^λ_t - V(S_t))를 하는데, 빼는 V(S_t)는 방금 고친 값이다.
    그래서 같은 칸을 여러 번 지나도 α ≤ 1이면 목표들 사이를 벗어나지 않는다.
    """
    V = np.full(n_states, v0, dtype=float)
    for ep in episodes:
        G = lambda_returns(ep, V, lam, gamma)
        for (s, *_), g in zip(ep, G):
            V[s] += alpha * (g - V[s])
        if record is not None:
            record.append(V.copy())
    return V


def td_lambda(episodes, n_states, lam, alpha, gamma=1.0, trace="accumulating", online=True, v0=0.0,
              record=None, trace_log=None, blowup=10.0):
    """적격 흔적으로 구현한 TD(λ) (후방 관점).

    e <- γλ e;  accumulating: e[s] += 1,  replacing: e[s] = 1
    δ = r + γ V(s') - V(s);  V <- V + α δ e  (online=False면 판 끝에 한꺼번에)
    |V|가 blowup을 넘거나 유한하지 않으면 발산으로 보고 멈춘다(반환값의 두 번째 원소 True).
    trace_log가 리스트면 걸음마다 (t, s, r, s2, done, e 사본)을 넣는다.
    """
    V = np.full(n_states, v0, dtype=float)
    for ep in episodes:
        e = np.zeros(n_states)
        dV = np.zeros(n_states)
        for t, (s, r, s2, done) in enumerate(ep):
            e *= gamma * lam
            if trace == "accumulating":
                e[s] += 1.0
            elif trace == "replacing":
                e[s] = 1.0
            else:
                raise ValueError(trace)
            delta = r + (0.0 if done else gamma * V[s2]) - V[s]
            if online:
                V += alpha * delta * e
            else:
                dV += alpha * delta * e
            if trace_log is not None:
                trace_log.append((t, s, r, s2, done, e.copy()))
        if not online:
            V += dV
        if record is not None:
            record.append(V.copy())
        if not np.all(np.isfinite(V)) or np.abs(V).max() > blowup:
            return V, True
    return V, False


# ---------------- 실험용 일괄 구현 ((시드 × 보폭) 행을 한꺼번에) ----------------

def pack_episode(eps_k):
    """시드별 k번째 판 B개를 (B, Tmax) 배열로 묶는다. 짧은 판은 mask로 가린다."""
    B = len(eps_k)
    T = np.array([len(ep) for ep in eps_k])
    Tm = int(T.max())
    S = np.zeros((B, Tm + 1), dtype=np.int64)
    R = np.zeros((B, Tm))
    D = np.zeros((B, Tm), dtype=bool)
    for b, ep in enumerate(eps_k):
        for t, (s, r, s2, d) in enumerate(ep):
            S[b, t], R[b, t], D[b, t] = s, r, d
        S[b, len(ep)] = ep[-1][2]
    M = np.arange(Tm)[None, :] < T[:, None]
    return S, R, D, M, T


def n_step_td_batch(episodes_by_seed, n_states, n, alphas, gamma=1.0, v0=0.0):
    """행 = (시드, 보폭). 반환: RMSE를 재기 위한 판별 V 기록 [K, B*A, n_states]. episodes_by_seed[b][k]."""
    B, K, A = len(episodes_by_seed), len(episodes_by_seed[0]), len(alphas)
    alpha = np.tile(np.asarray(alphas, dtype=float), B)            # 행 b*A + a
    V = np.full((B * A, n_states), v0)
    rows = np.arange(B * A)
    hist = np.zeros((K, B * A, n_states))
    for k in range(K):
        S, R, D, M, T = pack_episode([episodes_by_seed[b][k] for b in range(B)])
        S, R, M, T = np.repeat(S, A, 0), np.repeat(R, A, 0), np.repeat(M, A, 0), np.repeat(T, A)
        Tm = R.shape[1]
        disc = gamma ** np.arange(Tm)
        P = np.concatenate([np.zeros((B * A, 1)), np.cumsum(R * disc, axis=1)], axis=1)  # P[:, i] = Σ_{j<i} γ^j R_{j+1}
        for tau in range(Tm):
            valid = tau < T
            if not valid.any():
                break
            end = T if n is None else np.minimum(tau + n, T)
            G = (P[rows, end] - P[:, tau]) / gamma ** tau
            if n is not None:
                boot = (tau + n) < T
                idx = np.minimum(tau + n, Tm)
                G = G + np.where(boot, gamma ** n * V[rows, S[rows, idx]], 0.0)
            s = S[:, tau]
            upd = alpha * (G - V[rows, s])
            V[rows, s] += np.where(valid, upd, 0.0)
        hist[k] = V
    return hist


def td_lambda_batch(episodes_by_seed, n_states, lam, alphas, gamma=1.0, trace="accumulating", online=True,
                    v0=0.0, blowup=10.0):
    """td_lambda의 일괄 판. 발산한 행은 그 시점의 값으로 멈추고 diverged에 표시한다.

    반환: (hist [K, B*A, n_states], diverged [B*A] bool, diverged_at [B*A] 판 번호(1부터) 또는 0)
    """
    B, K, A = len(episodes_by_seed), len(episodes_by_seed[0]), len(alphas)
    alpha = np.tile(np.asarray(alphas, dtype=float), B)
    V = np.full((B * A, n_states), v0)
    rows = np.arange(B * A)
    alive = np.ones(B * A, dtype=bool)
    div_at = np.zeros(B * A, dtype=np.int64)
    hist = np.zeros((K, B * A, n_states))
    for k in range(K):
        S, R, D, M, T = pack_episode([episodes_by_seed[b][k] for b in range(B)])
        S, R, D, M = np.repeat(S, A, 0), np.repeat(R, A, 0), np.repeat(D, A, 0), np.repeat(M, A, 0)
        e = np.zeros((B * A, n_states))
        dV = np.zeros((B * A, n_states))
        for t in range(R.shape[1]):
            act = M[:, t] & alive
            s, s2 = S[:, t], S[:, t + 1]
            e *= gamma * lam
            if trace == "accumulating":
                e[rows, s] += act
            else:
                e[rows, s] = np.where(act, 1.0, e[rows, s])
            with np.errstate(over="ignore", invalid="ignore"):
                delta = R[:, t] + np.where(D[:, t], 0.0, gamma * V[rows, s2]) - V[rows, s]
                step = np.where(act, alpha * delta, 0.0)[:, None] * e
                if online:
                    V += step
                else:
                    dV += step
        if not online:
            V += dV
        with np.errstate(invalid="ignore"):
            bad = alive & (~np.all(np.isfinite(V), axis=1) | (np.abs(V).max(axis=1) > blowup))
        div_at[bad] = k + 1
        alive &= ~bad
        hist[k] = V
    return hist, div_at > 0, div_at


def offline_lambda_return_batch(episodes_by_seed, n_states, lam, alphas, gamma=1.0, v0=0.0, blowup=10.0):
    """offline_lambda_return의 일괄 판. 반환 꼴은 td_lambda_batch와 같다."""
    B, K, A = len(episodes_by_seed), len(episodes_by_seed[0]), len(alphas)
    alpha = np.tile(np.asarray(alphas, dtype=float), B)
    V = np.full((B * A, n_states), v0)
    rows = np.arange(B * A)
    div_at = np.zeros(B * A, dtype=np.int64)
    hist = np.zeros((K, B * A, n_states))
    for k in range(K):
        S, R, D, M, T = pack_episode([episodes_by_seed[b][k] for b in range(B)])
        S, R, D, M = np.repeat(S, A, 0), np.repeat(R, A, 0), np.repeat(D, A, 0), np.repeat(M, A, 0)
        Tm = R.shape[1]
        G = np.zeros((B * A, Tm))
        nxt = np.zeros(B * A)
        for t in range(Tm - 1, -1, -1):                      # 판 시작의 V로 λ-리턴을 뒤에서부터
            g = np.where(D[:, t], R[:, t], R[:, t] + gamma * ((1.0 - lam) * V[rows, S[:, t + 1]] + lam * nxt))
            G[:, t] = np.where(M[:, t], g, 0.0)
            nxt = G[:, t]
        for t in range(Tm):                                  # 판 끝에서 앞에서부터 차례로
            s = S[:, t]
            V[rows, s] += np.where(M[:, t], alpha * (G[:, t] - V[rows, s]), 0.0)
        bad = (div_at == 0) & (~np.all(np.isfinite(V), axis=1) | (np.abs(V).max(axis=1) > blowup))
        div_at[bad] = k + 1
        hist[k] = V
    return hist, div_at > 0, div_at
