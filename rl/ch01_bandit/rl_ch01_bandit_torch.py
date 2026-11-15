"""강화학습 1장 PyTorch 대응 구현: UCB1과 Thompson 선택을 텐서(float64)로.

난수는 NumPy에서 만들어 텐서로 넘긴다(동점 깨기 난수, Thompson의 베타 표본). 그래서 NumPy 구현과
비교되는 것은 표본추출 자체가 아니라 사후분포 갱신, UCB1 지수 계산, argmax와 동점 처리다.
같은 보상 표와 같은 rng 시드라면 선택 열이 NumPy 구현과 한 칸도 다르지 않아야 한다.

입력: policy_name("ucb1" 또는 "thompson"), NumPy 쪽 환경(보상 표 공유), 횟수, rng. 출력: 선택 기록 (S, n) 텐서.
탐험 파라미터는 NumPy 구현과 같다(UCB1 보너스 sqrt(2 ln n / n_j), Thompson 사전 Beta(1,1)). 할인율은 이 장에 없다.
"""

import numpy as np
import torch


# NumPy 판과 같은 논리의 동점 깨기(최댓값 칸 중 u로 균등 선택)
def argmax_random_tie(values, u):
    is_max = values == values.max(dim=1, keepdim=True).values
    cnt = is_max.sum(dim=1)
    pick = torch.minimum((u * cnt).long(), cnt - 1)
    return torch.argmax((torch.cumsum(is_max.long(), dim=1) > pick[:, None]).long(), dim=1)


def ucb1_index(N, X, n):
    # 식: xbar_j + sqrt(2 ln n / n_j). float64로 맞춰 NumPy와의 차이를 반올림 수준으로 묶는다. N=0 칸의 nan/inf는 아래 where가 덮어쓴다
    idx = X / N + torch.sqrt(2.0 * torch.log(torch.tensor(float(n), dtype=torch.float64)) / N)
    return torch.where(N == 0, torch.tensor(float("inf"), dtype=torch.float64), idx)


def beta_posterior(N, X, a0=1.0, b0=1.0):
    # 식: Beta(1+X, 1+N-X) (NumPy 판과 동일)
    return a0 + X, b0 + (N - X)


def run(policy_name, env, n_steps, rng):
    """policy_name: "ucb1" 또는 "thompson". env는 NumPy 쪽 BernoulliBandit(보상 표 공유)."""
    env.reset()
    S, K = env.S, env.K
    N = torch.zeros((S, K), dtype=torch.float64)
    X = torch.zeros((S, K), dtype=torch.float64)
    actions = torch.zeros((S, n_steps), dtype=torch.long)
    rows = torch.arange(S)
    # UCB1만 처음 K번을 손잡이 순서대로 당긴다(NumPy의 round_robin과 같음)
    round_robin = policy_name == "ucb1"
    for t in range(n_steps):
        if round_robin and t < K:
            a = torch.full((S,), t, dtype=torch.long)
        elif policy_name == "ucb1":
            u = torch.from_numpy(rng.random(S))
            # t = 지금까지 당긴 총횟수(식의 n). 난수 소비 순서(UCB1: 동점용 u 하나)가 NumPy 판과 같아야 선택 열이 일치한다
            a = argmax_random_tie(ucb1_index(N, X, t), u)
        elif policy_name == "thompson":
            # Thompson의 난수 소비 순서: 베타 표본 먼저, 그 다음 동점용 u (NumPy 판과 동일). 표본추출은 NumPy rng가 한다
            al, be = beta_posterior(N, X)
            theta = torch.from_numpy(rng.beta(al.numpy(), be.numpy()))
            u = torch.from_numpy(rng.random(S))
            a = argmax_random_tie(theta, u)
        else:
            raise ValueError(policy_name)
        # 환경은 NumPy 쪽 표를 공유(반복마다 같은 보상)
        r = torch.from_numpy(env.pull(a.numpy().astype(np.int64)))
        # N_a += 1, X_a += r
        N[rows, a] += 1
        X[rows, a] += r
        actions[:, t] = a
    return actions
