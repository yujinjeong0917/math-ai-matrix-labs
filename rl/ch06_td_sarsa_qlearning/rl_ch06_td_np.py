"""강화학습 6장 NumPy 최소 구현: 끝까지 기다리지 않고 "보상 + 다음 칸의 지금 추정값"으로 고친다.

5장 MC (상수 alpha)   V(s) <- V(s) + alpha [G_t - V(s)]                         판이 끝나야 G_t를 안다
6장 TD(0)             V(s) <- V(s) + alpha [r + gamma V(s') - V(s)]              한 걸음마다
    Sarsa             Q(s,a) <- Q(s,a) + alpha [r + gamma Q(s',a') - Q(s,a)]     a'는 실제로 고른 다음 행동
    Q-러닝            Q(s,a) <- Q(s,a) + alpha [r + gamma max_b Q(s',b) - Q(s,a)] 다음 행동과 상관없이 최댓값
    Expected Sarsa    Q(s,a) <- Q(s,a) + alpha [r + gamma sum_b pi(b|s') Q(s',b) - Q(s,a)]

종료 칸으로 들어가는 전이는 다음 추정값을 0으로 둔다(목표 = r).
속도를 위해 표는 파이썬 리스트로 들고 다니고, 끝에서 NumPy 배열로 돌려준다.
"""

import numpy as np

from rl_ch06_envs import Uniforms


# ---------- 예측: 정책이 정해진 경우 ----------

def td0_update(V, s, r, s2, done, alpha, gamma):
    """한 걸음 TD(0). 고친 뒤의 V[s]를 돌려준다."""
    # 식: 목표 = r + gamma V(s'). 종료로 들어가는 전이면 다음 가치를 0으로 두어 목표 = r
    target = r if done else r + gamma * V[s2]
    # 식: V(s) <- V(s) + alpha [목표 - V(s)]. 괄호 안이 TD 오차 delta
    V[s] = V[s] + alpha * (target - V[s])
    return V[s]


def td0(episodes, n_states, alpha, gamma, v0=0.0, record=None):
    """전이 목록의 목록 episodes = [[(s, r, s_next, done), ...], ...]로 TD(0)를 돌린다.

    record가 리스트면 판마다 V의 사본을 붙인다(학습 곡선용).
    """
    V = [float(v0)] * n_states
    for ep in episodes:
        for s, r, s2, done in ep:
            # 식: delta = r + gamma V(s') - V(s). 아래 두 줄이 V(s) <- V(s) + alpha delta. 목표는 고치기 전의 V[s2]로 계산한다
            target = r if done else r + gamma * V[s2]
            V[s] += alpha * (target - V[s])
        if record is not None:
            record.append(list(V))
    return np.array(V)


def mc_constant_alpha(episodes, n_states, alpha, gamma, v0=0.0, record=None, every_visit=True):
    """판이 끝난 뒤 G_t로 V(s) <- V(s) + alpha (G_t - V(s)). 5장의 증분형에서 1/N 대신 상수 alpha."""
    V = [float(v0)] * n_states
    for ep in episodes:
        g, G = 0.0, [0.0] * len(ep)
        for t in range(len(ep) - 1, -1, -1):
            # 식: G_t = r_t + gamma G_{t+1}. ep[t][1]이 그 전이의 보상 r
            g = ep[t][1] + gamma * g
            G[t] = g
        seen = set()
        for t, (s, _, _, _) in enumerate(ep):
            if not every_visit:
                if s in seen:
                    continue
                seen.add(s)
            # 식: V(s) <- V(s) + alpha (G_t - V(s)). 목표가 실제 남은 합 G_t라 판이 끝난 뒤에야 쓸 수 있다. every_visit=False면 첫 방문 t만
            V[s] += alpha * (G[t] - V[s])
        if record is not None:
            record.append(list(V))
    return np.array(V)


def mc_sample_average(episodes, n_states, gamma, v0=0.0, record=None):
    """5장의 첫 방문 MC(1/N 평균). 비교 기준용. 처음 방문하면 V = G가 되므로 v0는 아직 못 간 칸에만 남는다."""
    V, N = [float(v0)] * n_states, [0] * n_states
    for ep in episodes:
        g, G = 0.0, [0.0] * len(ep)
        for t in range(len(ep) - 1, -1, -1):
            # G_t 계산은 위와 같다
            g = ep[t][1] + gamma * g
            G[t] = g
        seen = set()
        for t, (s, _, _, _) in enumerate(ep):
            if s in seen:
                continue
            seen.add(s)
            N[s] += 1
            # 식: V <- V + (G - V) / N. 5장의 증분형 평균(상수 alpha 대신 1/N). N[s]를 먼저 올린 뒤 나눈다
            V[s] += (G[t] - V[s]) / N[s]
        if record is not None:
            record.append(list(V))
    return np.array(V)


# ---------- 제어: 행동가치 Q로 정책을 고친다 ----------

def greedy_set(q_row):
    """Q 한 행에서 최댓값과 같은(동률 포함) 행동 번호 목록."""
    m = max(q_row)
    return [a for a, x in enumerate(q_row) if x == m]


def eps_greedy_probs(q_row, eps):
    """epsilon-탐욕 정책의 확률. 동률인 최댓값 행동들이 (1 - eps)를 나눠 갖는다."""
    A = len(q_row)
    ties = greedy_set(q_row)
    # 식: pi(a|s) = eps/|A| + (1 - eps)/|동률 수| * 1[a가 최댓값 행동]. 합은 1
    p = [eps / A] * A
    for a in ties:
        p[a] += (1.0 - eps) / len(ties)
    return p


def eps_greedy(q_row, eps, uni, tie="random"):
    """eps 확률로 아무 행동, 아니면 최댓값 행동. 난수는 두 번 쓴다.

    tie="random"이면 동률 중 무작위, "first"면 번호가 가장 작은 행동(5장의 갇히는 실험과 같은 규칙).
    """
    # 분기와 관계없이 항상 난수 2개를 쓴다(난수 스트림의 위치가 일정)
    u1, u2 = uni(), uni()
    # 확률 eps로 탐험: u2로 행동 번호를 균등하게 고른다. min(...)은 u2가 1에 가까울 때 범위를 벗어나는 것을 막는다
    if u1 < eps:
        return min(int(u2 * len(q_row)), len(q_row) - 1)
    # 그 외에는 탐욕: 최댓값 동률 중 하나(random은 u2로, first는 가장 작은 번호)
    ties = greedy_set(q_row)
    if tie == "first":
        return ties[0]
    return ties[min(int(u2 * len(ties)), len(ties) - 1)]


def control(env, method, episodes, alpha, gamma, eps, seed, q0=0.0, max_steps=10_000, log=None, tie="random"):
    """절벽 격자에서 Sarsa / Q-러닝 / Expected Sarsa / MC 제어(첫 방문, 상수 alpha)를 돌린다.

    method: "sarsa", "q", "expected", "mc".
    반환 dict(Q[S,A], returns, falls, lengths, truncated). 리턴은 그 판에 실제로 받은 보상의 합(온라인 리턴).
    log가 리스트면 전이 (episode, t, s, a, r, s_next, done, fell)을 붙인다(재생·대조용).
    """
    uni = Uniforms(np.random.default_rng(seed))
    S, A = env.S, env.A
    # Q[s][a]: 파이썬 중첩 리스트(속도를 위해). 끝에서 NumPy 배열로 바꿔 돌려준다
    Q = [[float(q0)] * A for _ in range(S)]
    rets, falls, lens, truncs, updates = [], [], [], 0, 0
    for ep in range(episodes):
        s = env.start
        # Sarsa 규칙상 첫 행동을 판 시작에 미리 고른다. 이후 a는 이전 걸음이 고른 a'를 이어받는다
        a = eps_greedy(Q[s], eps, uni, tie)
        G, nf, traj = 0.0, 0, []
        done = False
        for t in range(max_steps):
            s2, r, done, fell = env.step(s, a)
            # G: 이 판에 실제로 받은 보상의 합(온라인 리턴, 할인 없음). 학습 목표의 G_t와 다르다. 절벽 보상도 들어간다
            G += r
            nf += fell
            if log is not None:
                log.append((ep, t, s, a, r, s2, done, fell))
            # MC 제어: 판이 끝날 때까지 (s, a, r)만 쌓고 Q는 판이 끝난 뒤에 고친다(아래 method == "mc" and done)
            if method == "mc":
                traj.append((s, a, r))
                if done:
                    break
                s, a = s2, eps_greedy(Q[s2], eps, uni, tie)
                continue
            if done:
                # 종료로 가는 걸음: 다음 가치를 0으로 둔다(목표 = r)
                target = r
                a2 = None
            elif method == "sarsa":
                # 식(Sarsa): 목표 = r + gamma Q(s', a'). a'는 다음에 실제로 할 행동(같은 epsilon-탐욕 정책)이라 온폴리시
                a2 = eps_greedy(Q[s2], eps, uni, tie)
                target = r + gamma * Q[s2][a2]
            elif method == "q":
                a2 = None  # Q-러닝·Expected Sarsa는 고친 뒤에 다음 행동을 고른다
                # 식(Q-러닝): 목표 = r + gamma max_b Q(s', b). 실제로 고를 a'와 무관하게 최댓값을 쓰므로 오프폴리시
                target = r + gamma * max(Q[s2])
            elif method == "expected":
                a2 = None
                # 식(Expected Sarsa): 목표 = r + gamma sum_b pi(b|s') Q(s', b). a'를 뽑지 않고 정책 확률로 가중 평균한다
                p = eps_greedy_probs(Q[s2], eps)
                target = r + gamma * sum(pb * qb for pb, qb in zip(p, Q[s2]))
            else:
                raise ValueError(method)
            # 식: Q(s,a) <- Q(s,a) + alpha [목표 - Q(s,a)]. 세 TD 방법이 이 줄을 공유하고 목표만 다르다. 괄호 안이 TD 오차
            Q[s][a] += alpha * (target - Q[s][a])
            updates += 1
            if done:
                break
            # Q-러닝·Expected Sarsa는 Q를 고친 뒤에 다음 행동을 고른다(Sarsa는 목표를 만들 때 이미 골랐다)
            if a2 is None:
                a2 = eps_greedy(Q[s2], eps, uni, tie)
            s, a = s2, a2
        if not done:
            truncs += 1
        if method == "mc" and done:  # 판이 끝났을 때만 리턴을 안다
            g, seen = 0.0, set()
            Gs = [0.0] * len(traj)
            for t in range(len(traj) - 1, -1, -1):
                # 식: G_t = r_t + gamma G_{t+1}. traj[t] = (s, a, r)
                g = traj[t][2] + gamma * g
                Gs[t] = g
            for t, (si, ai, _) in enumerate(traj):
                if (si, ai) in seen:
                    continue
                seen.add((si, ai))
                # 식: Q(s,a) <- Q(s,a) + alpha (G_t - Q(s,a)). (s,a)의 첫 방문만(seen), 학습률은 상수 alpha
                Q[si][ai] += alpha * (Gs[t] - Q[si][ai])
                updates += 1
        rets.append(G)
        falls.append(nf)
        lens.append(t + 1)
    return {"Q": np.array(Q), "returns": rets, "falls": falls, "lengths": lens,
            "truncated": truncs, "updates": updates}


def greedy_rollout(env, Q, max_steps=100):
    """epsilon = 0, 동률이면 번호가 가장 작은 행동. (리턴, 걸음 수, 도착 여부, 지나간 칸들)."""
    s, G, path = env.start, 0.0, [env.start]
    for t in range(max_steps):
        # 식: pi(s) = argmax_a Q(s,a). np.argmax는 동률이면 가장 작은 번호를 돌려준다
        a = int(np.argmax(Q[s]))
        s, r, done, _ = env.step(s, a)
        # 할인 없는 합(gamma = 1 설정에서는 리턴과 같다)
        G += r
        path.append(s)
        if done:
            return G, t + 1, True, path
    return G, max_steps, False, path


# ---------- 전이 기록 재생 (PyTorch 대조용) ----------

def replay(log, S, A, method, alpha, gamma, eps=0.1, q0=0.0):
    """control()이 남긴 전이 기록을 그대로 다시 넣어 Q를 만든다. Sarsa의 a'는 같은 판의 다음 줄에서 읽는다."""
    Q = [[float(q0)] * A for _ in range(S)]
    for i, (ep, t, s, a, r, s2, done, _) in enumerate(log):
        if done:
            target = r
        elif method == "sarsa":
            # Sarsa의 a'는 기록의 바로 다음 줄의 행동(열 3). 절벽 추락도 done = False라 판이 이어져 다음 줄이 같은 판의 다음 걸음이다
            a2 = log[i + 1][3]
            target = r + gamma * Q[s2][a2]
        elif method == "q":
            target = r + gamma * max(Q[s2])
        elif method == "expected":
            p = eps_greedy_probs(Q[s2], eps)
            target = r + gamma * sum(pb * qb for pb, qb in zip(p, Q[s2]))
        else:
            raise ValueError(method)
        Q[s][a] += alpha * (target - Q[s][a])
    return np.array(Q)


def q_learning_chain(n, gamma, alpha, episodes, eps, seed):
    """결정적 사슬에서 Q-러닝. 테스트용(고정점 = Q* 해석해)."""
    from rl_ch06_envs import chain_step
    uni = Uniforms(np.random.default_rng(seed))
    Q = [[0.0, 0.0] for _ in range(n)]
    for _ in range(episodes):
        s = 0
        for _ in range(10_000):
            a = eps_greedy(Q[s], eps, uni)
            s2, r, done = chain_step(s, a, n)
            # 식(Q-러닝): 목표 = r + gamma max_b Q(s', b), 종료면 r
            target = r if done else r + gamma * max(Q[s2])
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s2
    return np.array(Q)
