"""강화학습 5장 NumPy 최소 구현: 전이확률 없이 에피소드의 리턴을 평균 내서 가치를 구한다.

3·4장 (모델 기반 백업)   V(s) = sum_a pi(a|s) sum_s' P(s'|s,a) [r + gamma V(s')]
5장   (표본 평균)        V(s) ~ (G_1 + ... + G_N) / N,   증분형 V <- V + (G - V) / N

에이전트는 P를 읽지 않는다. 환경(make_env)만 P로 다음 칸을 뽑아 주고, 에이전트는 (s, a, r) 기록만 본다.
오프폴리시(행동 정책 b로 모은 기록으로 목표 정책 pi의 값을 구하기)는 중요도 비율로 고친다.
    rho = prod_t pi(a_t|s_t) / b(a_t|s_t)
    일반(ordinary) IS  sum rho_i G_i / N           평균으로 따지면 정확(불편), 분산이 클 수 있음
    가중(weighted) IS  sum rho_i G_i / sum rho_i   한쪽으로 치우침(편향), 분산 작음
"""

from bisect import bisect_right
from collections import namedtuple

import numpy as np

# states[t], actions[t], rewards[t]: t번째 걸음에서 있던 칸, 고른 행동, 그 행동 뒤에 받은 보상(길이 같음). done=False면 상한에 걸려 잘린 판
Episode = namedtuple("Episode", "states actions rewards done final", defaults=(None,))  # final: 마지막에 도착한 칸


def make_env(P, R, term):
    """P[S,A,S]를 칸·행동마다 누적확률 목록으로 바꿔 둔다(표본 뽑기용)."""
    S, A, _ = P.shape
    cdf, nxt = [], []
    for s in range(S):
        cs, ns = [], []
        for a in range(A):
            # 확률이 양수인 다음 칸만 남겨 칸·행동마다 누적확률표(cdf)를 만든다
            idx = np.flatnonzero(P[s, a] > 0)
            c = np.cumsum(P[s, a, idx])
            # 부동소수 오차로 마지막 값이 1에 못 미쳐도 bisect가 범위를 벗어나지 않게 1로 고정
            c[-1] = 1.0
            cs.append(c.tolist())
            ns.append(idx.tolist())
        cdf.append(cs)
        nxt.append(ns)
    return {"cdf": cdf, "nxt": nxt, "R": R, "term": term.tolist(), "S": S, "A": A}


class _Uniforms:
    """rng.random()을 한 번에 4,096개씩 뽑아 두고 하나씩 꺼낸다(파이썬 반복이 빨라짐)."""

    def __init__(self, rng, block=4096):
        self.rng, self.block = rng, block
        self.buf, self.i = rng.random(block).tolist(), 0

    def __call__(self):
        if self.i == self.block:
            self.buf, self.i = self.rng.random(self.block).tolist(), 0
        u = self.buf[self.i]
        self.i += 1
        return u


def policy_cdf(policy):
    """정책 policy[S,A]의 칸별 누적확률표(리스트). 행동 표본 뽑기용이고 마지막 값은 1로 고정한다."""
    c = np.cumsum(policy, axis=1)
    c[:, -1] = 1.0
    return c.tolist()


def generate_episode(env, policy, s0, rng, max_steps=10_000, uni=None, pcdf=None):
    """정책 policy[S,A]로 s0에서 한 판을 끝까지(또는 max_steps까지) 걷는다.

    반환 Episode(states, actions, rewards, done). done=False면 상한에 걸려 잘린 판이라 리턴을 모른다.
    """
    uni = uni or _Uniforms(rng)
    pc = pcdf or policy_cdf(policy)
    cdf, nxt, R, term = env["cdf"], env["nxt"], env["R"], env["term"]
    s = s0
    S_, A_, R_ = [], [], []
    for _ in range(max_steps):
        if term[s]:
            return Episode(S_, A_, R_, True, s)
        # 역변환 표본추출: u ~ U[0,1)이 누적확률표의 어느 구간에 떨어지는지 찾으면 a ~ pi(.|s)
        a = bisect_right(pc[s], uni())
        # 같은 방식으로 s' ~ P(.|s,a). 이 줄은 환경이 하는 일이고 에이전트는 결과(s', r)만 본다. 걸음마다 난수 2개를 쓴다
        j = bisect_right(cdf[s][a], uni())
        s2 = nxt[s][a][min(j, len(nxt[s][a]) - 1)]
        S_.append(s)
        A_.append(a)
        # r = R[s,a,s']: (s,a) 뒤에 받은 보상이라 rewards[t]는 states[t], actions[t]와 같은 t
        R_.append(float(R[s, a, s2]))
        s = s2
    # 상한 max_steps에 걸려 나온 경우. 마지막 걸음에서 막 종료 칸에 들어갔다면 term[s]가 True라 done=True
    return Episode(S_, A_, R_, term[s], s)


def returns(rewards, gamma):
    """G_t = r_t + gamma G_{t+1}를 끝에서부터 계산한다. 한 판에 O(T)."""
    G = np.zeros(len(rewards))
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        # 식: G_t = r_t + gamma G_{t+1}. g=0에서 시작하므로 마지막 걸음은 G = r_T
        # rewards[t]는 (s_t, a_t) 뒤 보상(Sutton·Barto 표기의 R_{t+1})이라 G[0]이 판 전체의 할인 합
        g = rewards[t] + gamma * g
        G[t] = g
    return G


def _mc(episodes, gamma, S, first_visit):
    V = np.zeros(S)
    N = np.zeros(S, dtype=np.int64)
    skipped = 0
    for ep in episodes:
        if not ep.done:  # 끝나지 않은 판은 리턴을 모르니 아무것도 고칠 수 없다
            skipped += 1
            continue
        # G[t]: t번째 걸음 이후의 할인 리턴. 판이 끝나야 계산할 수 있다
        G = returns(ep.rewards, gamma)
        # 첫 방문이면 상태 s가 판 안에서 처음 나온 시각 t만 모으고, 모든 방문이면 모든 t를 쓴다
        if first_visit:
            seen = set()
            ts = []
            for t, s in enumerate(ep.states):
                if s not in seen:
                    seen.add(s)
                    ts.append(t)
        else:
            ts = range(len(ep.states))
        for t in ts:
            s = ep.states[t]
            # 식: V_N(s) = V_{N-1}(s) + (G_t - V_{N-1}(s)) / N, 즉 방문 횟수 N으로 나누는 표본 평균
            # N[s]를 먼저 올린 뒤 나누므로 첫 방문에서는 V = G가 된다(초기 0은 흔적이 남지 않음)
            N[s] += 1
            V[s] += (G[t] - V[s]) / N[s]  # 증분형 평균: V <- V + (G - V) / N
    return V, N, skipped


def mc_first_visit(episodes, gamma, S):
    """판마다 각 상태를 처음 방문한 시점의 리턴만 평균. 반환 (V, N, 건너뛴 판 수)."""
    return _mc(episodes, gamma, S, True)


def mc_every_visit(episodes, gamma, S):
    """판 안에서 같은 상태를 여러 번 방문하면 그때마다의 리턴을 모두 평균."""
    return _mc(episodes, gamma, S, False)


def eps_greedy(Q, eps):
    """Q[S,A]에서 epsilon-탐욕 정책 pi[S,A]를 만든다. 탐욕 행동 확률 1 - eps + eps/A, 나머지 eps/A."""
    S, A = Q.shape
    # 식: pi(a|s) = eps/|A| + (1 - eps) * 1[a = argmax_b Q(s,b)]. 먼저 모든 행동에 eps/|A|씩 깐다
    pi = np.full((S, A), eps / A)
    # 탐욕 행동 하나에 (1 - eps)를 더한다. 행의 합은 1. eps=0이면 결정적 정책, eps>0이면 모든 행동 확률이 양수(epsilon-soft)
    pi[np.arange(S), Q.argmax(axis=1)] += 1.0 - eps  # 동률이면 번호가 가장 작은 행동
    return pi


def mc_control_eps_soft(env, s0, gamma, eps, n_episodes, rng, max_steps=10_000, exploring_starts=False):
    """온폴리시 첫 방문 MC 제어(epsilon-soft). Q는 0에서 시작, 판이 끝날 때마다 고친다.

    exploring_starts=True면 판마다 종료 칸이 아닌 칸 하나와 첫 행동을 무작위로 골라 시작한다
    (Sutton·Barto 5.3절의 "exploring starts" 가정). 반환 Q, 마지막 정책, 판 길이 목록, 잘린 판 수.
    """
    S, A = env["S"], env["A"]
    # Q 초기값 0. 모두 동률이면 argmax가 행동 0(위)을 골라, eps=0에서는 첫 판이 끝나지 않고 갇힐 수 있다(실험 E1 c)
    Q = np.zeros((S, A))
    N = np.zeros((S, A), dtype=np.int64)
    uni = _Uniforms(rng)
    starts = [s for s in range(S) if not env["term"][s]]
    lengths, truncated = [], 0
    for _ in range(n_episodes):
        # 판마다 지금의 Q에서 epsilon-탐욕 정책을 새로 만든다. 이 정책으로 모은 판으로 같은 정책의 Q를 고치므로 온폴리시
        pi = eps_greedy(Q, eps)
        if exploring_starts:
            # exploring starts: 종료 칸이 아닌 칸을 균등하게 뽑아 시작 칸으로, 첫 행동도 균등하게
            st = starts[int(uni() * len(starts))]
            a0 = int(uni() * A)
            ep = _first_forced(env, pi, st, a0, rng, max_steps, uni)
        else:
            ep = generate_episode(env, pi, s0, rng, max_steps, uni=uni)
        lengths.append(len(ep.states))
        # 끝나지 않은 판은 리턴을 모르므로 Q를 고치지 않는다(길이만 기록)
        if not ep.done:
            truncated += 1
            continue
        G = returns(ep.rewards, gamma)
        # (s,a) 쌍의 첫 방문만 쓴다(첫 방문 MC). 한 판에서 같은 쌍이 다시 나오면 건너뜀
        seen = set()
        for t, (s, a) in enumerate(zip(ep.states, ep.actions)):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            N[s, a] += 1
            # 식: Q(s,a) <- Q(s,a) + (G_t - Q(s,a)) / N(s,a). 칸이 아니라 (칸, 행동) 쌍마다 표본 평균
            Q[s, a] += (G[t] - Q[s, a]) / N[s, a]
    # 반환하는 정책은 최종 Q에서 다시 만든 epsilon-탐욕 정책
    return Q, eps_greedy(Q, eps), lengths, truncated


def _first_forced(env, pi, s0, a0, rng, max_steps, uni):
    """첫 행동만 a0로 정하고 그 뒤는 pi로 걷는다(exploring starts용)."""
    cdf, nxt, R = env["cdf"], env["nxt"], env["R"]
    j = bisect_right(cdf[s0][a0], uni())
    s1 = nxt[s0][a0][min(j, len(nxt[s0][a0]) - 1)]
    # 첫 걸음을 이미 썼으므로 남은 상한은 max_steps - 1. 이어 붙인 판은 states[0] = s0, actions[0] = a0
    rest = generate_episode(env, pi, s1, rng, max_steps - 1, uni=uni)
    return Episode([s0] + rest.states, [a0] + rest.actions, [float(R[s0, a0, s1])] + rest.rewards, rest.done, rest.final)


def importance_ratio(ep, pi, b):
    """rho = prod_t pi(a_t|s_t) / b(a_t|s_t). 전이확률은 분자·분모에서 지워져 들어가지 않는다."""
    rho = 1.0
    for s, a in zip(ep.states, ep.actions):
        # 식: rho = prod_t pi(a_t|s_t) / b(a_t|s_t). b[s,a] > 0이어야 한다(b가 pi가 고르는 행동을 모두 덮어야 함)
        # pi가 0인 행동이 한 번이라도 나오면 rho = 0이 된다
        rho *= pi[s, a] / b[s, a]
    return rho


def off_policy_is(episodes, pi, b, gamma, weighted):
    """행동 정책 b로 모은 판들(모두 같은 시작 칸)로 목표 정책 pi의 시작 칸 가치를 추정한다.

    시작 칸의 첫 방문(t=0)만 쓴다. weighted=False면 일반 IS, True면 가중 IS(분모가 0이면 0).
    """
    num, den, n = 0.0, 0.0, 0
    for ep in episodes:
        if not ep.done:
            continue
        rho = importance_ratio(ep, pi, b)
        # G0: 시작 칸(t=0)의 할인 리턴. 시작 칸이 곧 종료 칸이라 걸음이 없으면 0
        G0 = returns(ep.rewards, gamma)[0] if ep.rewards else 0.0
        # 일반 IS의 분자 sum rho_i G_i와 가중 IS의 분모 sum rho_i를 함께 모은다(두 방법이 같은 rho를 씀)
        num += rho * G0
        den += rho
        n += 1
    # 식: 가중 IS = sum rho_i G_i / sum rho_i (분모가 0이면 0으로 둠)
    if weighted:
        return num / den if den > 0 else 0.0
    # 식: 일반 IS = sum rho_i G_i / N (N은 끝난 판 수). b = pi면 rho = 1이라 두 방법 모두 G0의 평균과 같다
    return num / n


def random_walk_lengths(n, rng, nbr, term, s0, cap):
    """균등 무작위 정책을 판 n개 동시에 걷게 해 판 길이를 센다(상한 cap). 잘린 판은 cap으로 둔다."""
    # 판 n개를 한꺼번에 한 걸음씩 진행. pos[i]는 i번째 판의 현재 칸이고 nbr[s,a]는 미끄러짐 없는 격자의 다음 칸 표
    pos = np.full(n, s0)
    length = np.zeros(n, dtype=np.int64)
    alive = ~term[pos]
    for _ in range(cap):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        # 균등 무작위 정책 pi(a|s) = 1/4: 아직 안 끝난 판마다 행동 4개 중 하나를 같은 확률로
        a = rng.integers(0, 4, size=idx.size)
        pos[idx] = nbr[pos[idx], a]
        length[idx] += 1
        # 종료 칸에 들어간 판은 alive=False로 빼서 더 걷지 않는다
        alive[idx] = ~term[pos[idx]]
    return length, alive  # alive=True면 cap에서 잘림



# ---- 실패 2용: 상태 하나짜리 문제를 벡터로 빠르게 ----

def sample_more_counts(n, rng, b_more=0.5):
    """행동 정책이 "한 번 더"를 b_more로 고를 때, 한 판에서 "한 번 더"를 고른 횟수 k를 n판 뽑는다.

    P(k) = b_more^k (1 - b_more). numpy의 geometric은 1부터 세므로 1을 뺀다.
    """
    return rng.geometric(1.0 - b_more, size=n) - 1


def rho_of_k(k, p_more, b_more=0.5):
    """k번 "한 번 더" 뒤 "그만"인 판의 중요도 비율 (p/b)^k * (1-p)/(1-b)."""
    # 식: rho_k = (p/b)^k * (1-p)/(1-b). 이 판은 "한 번 더"를 k번(걸음마다 비율 p/b) 고른 뒤 "그만"을 한 번 골랐다
    # 전이가 결정적이라 비율에 전이확률은 들어가지 않는다
    return np.power(p_more / b_more, k) * ((1.0 - p_more) / (1.0 - b_more))


def is_estimates_prefix(k, p_more, Ns, b_more=0.5):
    """같은 판 묶음의 앞 N판으로 낸 일반·가중 IS 추정값(리턴 G = k)."""
    rho = rho_of_k(k, p_more, b_more)
    # 이 문제는 G = k(gamma = 1). cs_num[i] = 앞 i+1판의 sum rho G, cs_den[i] = 앞 i+1판의 sum rho
    # 판 묶음 하나로 N을 키워 가며 추정값을 읽을 수 있다
    cs_num = np.cumsum(rho * k)
    cs_den = np.cumsum(rho)
    # 함정: 앞 N판의 합은 인덱스 N-1에 있다(0부터 센다)
    idx = np.asarray(Ns) - 1
    # 식: 일반 IS = cs_num / N, 가중 IS = cs_num / cs_den
    return cs_num[idx] / np.asarray(Ns), cs_num[idx] / cs_den[idx]


def one_state_truth(p_more):
    """목표 정책의 참값 v_pi = E[k] = p / (1 - p)."""
    # 식: v_pi = E_pi[k] = sum_k k p^k (1-p) = p / (1-p) (기하분포의 평균)
    return p_more / (1.0 - p_more)


def is_second_moment(p_more, b_more=0.5):
    """E_b[(rho G)^2] = sum_k b^k (1-b) (p/b)^(2k) ((1-p)/(1-b))^2 k^2. x = p^2/b >= 1이면 무한대."""
    # 식: x = p^2 / b. 급수 sum_k k^2 x^k = x(1+x) / (1-x)^3 이 x < 1일 때만 수렴한다
    x = p_more**2 / b_more
    # x >= 1이면 급수가 발산해 분산이 무한대. 일반 IS가 불편이어도 N을 키운 추정값이 매우 들쭉날쭉하다
    if x >= 1.0:
        return float("inf")
    # 급수 앞의 상수 c = (1-b) ((1-p)/(1-b))^2. 2차 모멘트 = c x (1+x) / (1-x)^3 (k = 0 항은 k^2 = 0이라 빠짐)
    c = (1.0 - b_more) * ((1.0 - p_more) / (1.0 - b_more)) ** 2
    return c * x * (1.0 + x) / (1.0 - x) ** 3
