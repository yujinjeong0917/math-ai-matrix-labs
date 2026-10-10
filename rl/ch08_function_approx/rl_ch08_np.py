"""강화학습 8장 NumPy 최소 구현: 선형 함수 근사와 반경사 TD(0), 그리고 그것이 발산하는 조건.

상태 번호는 0부터 센다. Baird(1995)의 별 문제에서 원 논문의 상태 1~6은 여기서 0~5,
가중치 w0~w6은 그대로 w[0]~w[6]이다.

- baird_star: 6상태 별 문제. V(i) = w0 + 2 w_i (i = 1..5), V(6) = 2 w0 + w6. 모든 보상 0.
  평가하려는 정책(목표 정책)에서는 모든 상태가 상태 6으로 간다(6은 6으로).
- behavior_chain: Baird 같은 논문의 Q-러닝 판에서 쓴 행동 정책을 상태 전이로만 옮긴 것.
  확률 1/6로 상태 6으로, 5/6로 1~5 중 하나로 균등하게 간다. 결과적으로 모든 행이 6개 상태에 1/6씩이다.
- hall: Baird 그림 2의 복도 문제. 표(가중치 = 상태 값), i -> i+1, 마지막 상태는 자기 자신으로.
- semi_gradient_td_sweeps: 전이 목록을 한 바퀴(스윕)씩 돈다. epoch=True면 Baird 식 (7)처럼
  한 바퀴 동안 고칠 양을 모았다가 한 번에 더하고, False면 전이마다 바로 고친다.
- residual_sweeps: Baird 식 (12)·(13)의 잔차 알고리즘. mix=1이면 잔차 경사, mix=0이면 직접(반경사) TD.
- regression_sweeps: 부트스트랩을 뺀 판. 목표로 참값(여기서는 실제 리턴의 기댓값)을 쓴다.
- td0_trajectory: 실제로 한 줄의 궤적을 따라가며 고친다. off_policy=True면 목표 정책과 행동 정책의
  확률 비(중요도 비) rho를 곱해 목표 정책의 값을 배우려 한다.
- expected_update_matrix: 기대 갱신을 w <- w + alpha (b - A w)로 쓸 때의 A = Phi^T D (I - gamma P) Phi.
"""

import numpy as np

N_STATES = 6
N_WEIGHTS = 7


# ---------------- 문제 정의 ----------------

def baird_star():
    """(Phi, next_state, R). Phi: (6, 7) 특징 행렬, next_state[s]: 목표 정책에서의 다음 상태."""
    Phi = np.zeros((N_STATES, N_WEIGHTS))
    for i in range(5):
        Phi[i, 0] = 1.0          # 모두가 함께 쓰는 손잡이 w0
        Phi[i, i + 1] = 2.0      # 자기 손잡이 w1..w5
    Phi[5, 0] = 2.0
    Phi[5, 6] = 1.0
    next_state = np.full(N_STATES, 5)
    R = np.zeros(N_STATES)
    return Phi, next_state, R


def table_features(n=N_STATES):
    """표: 상태마다 자기 칸 하나. 별 문제에서 w0을 빼고 상태 6의 손잡이를 w6 하나로 둔 것과 같은 표현력."""
    return np.eye(n)


def transition_matrix(next_state):
    n = len(next_state)
    P = np.zeros((n, n))
    P[np.arange(n), next_state] = 1.0
    return P


def behavior_chain():
    """행동 정책의 전이 행렬. 1/6 확률로 상태 6, 5/6 확률로 1~5 중 균등 -> 모든 칸이 1/6."""
    return np.full((N_STATES, N_STATES), 1.0 / N_STATES)


def stationary_distribution(P):
    vals, vecs = np.linalg.eig(P.T)
    k = int(np.argmin(np.abs(vals - 1.0)))
    d = np.real(vecs[:, k])
    return d / d.sum()


def hall(n=6):
    """Baird 그림 2. V(i) = w_i, i -> i+1, 마지막 상태는 자기 자신으로. 모든 보상 0."""
    nxt = np.minimum(np.arange(n) + 1, n - 1)
    return np.eye(n), nxt, np.zeros(n)


# ---------------- 기대 갱신 분석 ----------------

def expected_update_matrix(Phi, P, D, gamma):
    """A = Phi^T D (I - gamma P) Phi. 기대 갱신은 w <- w + alpha (b - A w)."""
    D = np.diag(D) if np.ndim(D) == 1 else D
    n = Phi.shape[0]
    return Phi.T @ D @ (np.eye(n) - gamma * P) @ Phi


def residual_update_matrix(Phi, P, D, gamma, mix=1.0):
    """잔차 알고리즘(Baird 식 12)의 기대 갱신 행렬 M. w <- w - alpha M w (보상 0일 때).

    한 전이의 갱신: dw = -alpha * delta * (mix * gamma * phi(s') - phi(s)),
    delta = r + gamma phi(s')w - phi(s)w. 보상이 0이면 dw = -alpha (phi(s) - mix gamma phi(s')) (phi(s) - gamma phi(s'))^T w.
    mix = 0이면 expected_update_matrix와 같다.
    """
    D = np.diag(D) if np.ndim(D) == 1 else D
    left = Phi - mix * gamma * (P @ Phi)
    right = Phi - gamma * (P @ Phi)
    return left.T @ D @ right


def bellman_error(w, Phi, P, R, gamma, D):
    """평균 제곱 벨만 잔차의 제곱근. sqrt(sum_s D_s (R_s + gamma (P Phi w)_s - (Phi w)_s)^2)."""
    v = Phi @ w
    r = R + gamma * (P @ v) - v
    return float(np.sqrt(np.sum(np.asarray(D) * r ** 2)))


def value_norm(w, Phi):
    """참값이 모두 0인 문제에서의 값 오차: 상태 값의 최대 절댓값."""
    return float(np.max(np.abs(Phi @ w)))


# ---------------- 스윕 단위 학습 ----------------

def _run_sweeps(step_fn, w0, sweeps, record_every=None, blowup=1e8):
    """step_fn(w) -> 새 w. 발산(노름 > blowup 또는 유한하지 않음)하면 멈춘다."""
    w = np.array(w0, dtype=float)
    hist = [(0, w.copy())] if record_every else []
    diverged_at = None
    for k in range(1, sweeps + 1):
        w = step_fn(w)
        if not np.all(np.isfinite(w)) or np.linalg.norm(w) > blowup:
            diverged_at = k
            if record_every:
                hist.append((k, w.copy()))
            break
        if record_every and k % record_every == 0:
            hist.append((k, w.copy()))
    return w, hist, diverged_at


def residual_sweeps(Phi, next_state, R, w0, alpha, gamma, sweeps, mix=0.0, epoch=True,
                    record_every=None, blowup=1e8):
    """전이 (s -> next_state[s], 보상 R[s])를 상태 순서대로 한 번씩 보는 스윕을 반복한다.

    mix = 0: 직접(반경사) TD(0). Baird 식 (3)·(7).  dw = alpha * delta * phi(s)
    mix = 1: 잔차 경사. Baird 식 (6)·(8).         dw = -alpha * delta * (gamma phi(s') - phi(s))
    그 사이: 잔차 알고리즘. Baird 식 (12)·(13).
    epoch=True면 한 스윕 동안의 dw를 모아 한 번에 더한다(Baird 식 7·8의 epoch-wise).
    """
    n = Phi.shape[0]

    def step(w):
        if epoch:
            v = Phi @ w
            delta = R + gamma * v[next_state] - v                  # (n,)
            direction = Phi - mix * gamma * Phi[next_state]       # (n, d): 각 행 = phi(s) - mix gamma phi(s')
            return w + alpha * direction.T @ delta
        for s in range(n):
            s2 = next_state[s]
            delta = R[s] + gamma * Phi[s2] @ w - Phi[s] @ w
            w = w + alpha * delta * (Phi[s] - mix * gamma * Phi[s2])
        return w

    return _run_sweeps(step, w0, sweeps, record_every, blowup)


def semi_gradient_td_sweeps(Phi, next_state, R, w0, alpha, gamma, sweeps, epoch=True,
                            record_every=None, blowup=1e8):
    """반경사 TD(0). 목표 r + gamma V(s')는 상수처럼 두고 V(s)의 기울기 phi(s) 방향으로만 고친다."""
    return residual_sweeps(Phi, next_state, R, w0, alpha, gamma, sweeps, mix=0.0, epoch=epoch,
                           record_every=record_every, blowup=blowup)


def regression_sweeps(Phi, v_target, w0, alpha, sweeps, record_every=None, blowup=1e8):
    """부트스트랩 없음: 목표를 참값 v_target으로 고정한 회귀. dw = alpha * sum_s (v_target_s - phi(s)w) phi(s)."""
    def step(w):
        return w + alpha * Phi.T @ (v_target - Phi @ w)
    return _run_sweeps(step, w0, sweeps, record_every, blowup)


# ---------------- 궤적 위의 학습 ----------------

def td0_trajectory(Phi, P_behavior, P_target, R, w0, alpha, gamma, steps, seed,
                   off_policy=True, s0=None, log=None, log_steps=0, blowup=1e8):
    """행동 정책 P_behavior로 궤적 하나를 만들며 TD(0)으로 고친다.

    off_policy=True: 목표 정책 P_target의 값을 배운다. 전이마다 rho = P_target[s,s'] / P_behavior[s,s'].
                     dw = alpha * rho * delta * phi(s)  (반경사 off-policy TD(0))
    off_policy=False: 행동 정책 자신의 값을 배운다(온폴리시). rho = 1.
    보상은 상태에만 달려 있다고 둔다(R[s]).
    log: 리스트를 주면 처음 log_steps 걸음을 공통 스키마 딕셔너리로 쌓는다.
    """
    rng = np.random.default_rng(seed)
    n = Phi.shape[0]
    w = np.array(w0, dtype=float)
    s = int(rng.integers(n)) if s0 is None else int(s0)
    diverged_at = None
    norms = []
    for t in range(steps):
        s2 = int(rng.choice(n, p=P_behavior[s]))
        rho = P_target[s, s2] / P_behavior[s, s2] if off_policy else 1.0
        delta = R[s] + gamma * Phi[s2] @ w - Phi[s] @ w
        w = w + alpha * rho * delta * Phi[s]
        if log is not None and t < log_steps:
            log.append({"seed": seed, "episode": 0, "t": t, "s": s + 1, "a": f"to_{s2 + 1}",
                        "r": float(R[s]), "s_next": s2 + 1, "done": False,
                        "info": {"rho": float(rho), "delta": float(delta),
                                 "w": [float(x) for x in w], "w_norm": float(np.linalg.norm(w))}})
        if not np.all(np.isfinite(w)) or np.linalg.norm(w) > blowup:
            diverged_at = t + 1
            break
        if (t + 1) % 1000 == 0:
            norms.append(float(np.linalg.norm(w)))
        s = s2
    return w, diverged_at, norms
