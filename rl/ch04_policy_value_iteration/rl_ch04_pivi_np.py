"""강화학습 4장 NumPy 최소 구현: 정책 반복, 가치 반복, 수정 정책 반복.

Q(s,a)   = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]          한 걸음 앞을 내다본 값
개선     pi'(s) = argmax_a Q^pi(s,a)                             정책 개선 정리: V^{pi'} >= V^pi
정책 반복 평가(정확히) -> 개선 -> 평가 -> ... 정책이 안 바뀌면 멈춤
가치 반복 V_{k+1}(s) = max_a Q_k(s,a)  (= T V_k)                 평가를 한 스윕으로 줄여 개선과 합침
수정 정책 반복(k) 개선 한 번 뒤 평가 스윕을 k번만                  k=0이면 가치 반복, k->무한이면 정책 반복
축약     ||T V - T U||_inf <= gamma ||V - U||_inf

연산 수 세기("백업"): 한 상태에서 한 행동의 기댓값 sum_s' P(s'|s,a)[...]를 한 번 계산하면 백업 1번.
  가치 반복 한 스윕 = S*A, 정책 평가 한 스윕 = S, 개선 한 번 = S*A. 직접 풀이(solve)는 따로 약 (2/3)S^3 FLOP.
"""

import numpy as np

TIE_TOL = 1e-9


def q_from_v(V, P, R, gamma):
    """Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]."""
    return np.einsum("sap,sap->sa", P, R + gamma * V[None, None, :])


def bellman_T(V, P, R, gamma):
    """벨만 최적 연산자 T. 가치 반복의 한 스윕."""
    return q_from_v(V, P, R, gamma).max(axis=1)


def greedy_policy(V, P, R, gamma, tie="first", rng=None, old=None, tol=TIE_TOL):
    """V에 대해 한 걸음 앞을 보고 가장 좋은 행동을 고른다. 반환: 행동 번호 배열 pi[S].

    tie: 최댓값과 tol 안으로 같은 행동이 여럿일 때
      "first"  번호가 가장 작은 행동 (결정론적)
      "random" 그중 무작위 (rng 필요)  -> 정책 반복이 끝나지 않을 수 있다(4장 실패 1)
      "keep"   지금 행동(old)이 그중에 있으면 그대로, 없으면 "first"   (Sutton·Barto 연습 4.4의 고침 방향)
      "raw"    np.argmax 그대로. 동률 판정 없이 부동소수 잡음이 승자를 정한다
    """
    Q = q_from_v(V, P, R, gamma)
    if tie == "raw":
        return Q.argmax(axis=1)
    best = Q >= Q.max(axis=1, keepdims=True) - tol
    pi = best.argmax(axis=1)  # 첫 번째 True
    if tie == "random":
        for s in np.flatnonzero(best.sum(axis=1) > 1):
            pi[s] = rng.choice(np.flatnonzero(best[s]))
    elif tie == "keep" and old is not None:
        keep = best[np.arange(len(old)), old]
        pi[keep] = old[keep]
    return pi


def policy_mats(pi, P, R):
    """결정론적 정책 pi[S]에서 P^pi[S,S], r^pi[S]."""
    idx = np.arange(P.shape[0])
    Ppi = P[idx, pi]
    rpi = (Ppi * R[idx, pi]).sum(axis=1)
    return Ppi, rpi


def evaluate_exact(pi, P, R, gamma):
    """3장의 직접 풀이: (I - gamma P^pi) V = r^pi."""
    Ppi, rpi = policy_mats(pi, P, R)
    return np.linalg.solve(np.eye(P.shape[0]) - gamma * Ppi, rpi)


def evaluate_sweeps(pi, P, R, gamma, V, k):
    """3장의 반복 평가를 k스윕만(두 배열, 야코비식). 백업 수 = k*S."""
    Ppi, rpi = policy_mats(pi, P, R)
    for _ in range(k):
        V = rpi + gamma * Ppi @ V
    return V


def policy_iteration(P, R, gamma, pi0=None, tie="first", rng=None, max_iter=1000):
    """평가(직접 풀이) -> 개선을 정책이 안 바뀔 때까지.

    반환: (pi, V, history, converged). history[i] = {"changed": 바뀐 칸 수, "V": 평가값}. 끝나지 않으면 converged=False.
    """
    S = P.shape[0]
    pi = np.zeros(S, dtype=np.int64) if pi0 is None else np.asarray(pi0).copy()
    history = []
    for it in range(1, max_iter + 1):
        V = evaluate_exact(pi, P, R, gamma)
        new = greedy_policy(V, P, R, gamma, tie=tie, rng=rng, old=pi)
        changed = int((new != pi).sum())
        history.append({"changed": changed, "V": V})
        pi = new
        if changed == 0:
            return pi, V, history, True
    return pi, V, history, False


def value_iteration(P, R, gamma, theta=1e-10, V0=None, max_iter=1_000_000, V_star=None):
    """V_{k+1} = T V_k 를 ||V_{k+1} - V_k||_inf < theta 까지.

    반환: (V, history). history[k-1] = (||V_k - V_{k-1}||, ||V_k - V*|| 또는 None).
    """
    V = np.zeros(P.shape[0]) if V0 is None else V0.copy()
    history = []
    for _ in range(max_iter):
        V_new = bellman_T(V, P, R, gamma)
        d = float(np.max(np.abs(V_new - V)))
        e = None if V_star is None else float(np.max(np.abs(V_new - V_star)))
        history.append((d, e))
        V = V_new
        if d < theta:
            return V, history
    return V, history


def modified_policy_iteration(P, R, gamma, k, theta=1e-10, max_iter=1_000_000):
    """개선 한 번(=가치 반복 한 스윕, 그 결과의 탐욕 정책) 뒤 그 정책으로 평가 스윕 k번.

    k=0이면 가치 반복과 같다. 멈춤: 개선 스윕에서 ||T V - V|| < theta.
    반환: (pi, V, iters, backups). backups = 개선 S*A + 평가 k*S 를 반복마다 더한 값.
    """
    S, A = P.shape[0], P.shape[1]
    V = np.zeros(S)
    backups = 0
    for it in range(1, max_iter + 1):
        Q = q_from_v(V, P, R, gamma)
        TV, pi = Q.max(axis=1), Q.argmax(axis=1)
        backups += S * A
        d = float(np.max(np.abs(TV - V)))
        V = evaluate_sweeps(pi, P, R, gamma, TV, k)
        backups += k * S
        if d < theta:
            return pi, V, it, backups
    raise RuntimeError("max_iter 안에 멈추지 않았어요")


def theory_vi_sweeps(eps, gamma, rmax=1.0):
    """V0=0에서 ||V0 - V*|| <= rmax/(1-gamma) 이므로 gamma^k rmax/(1-gamma) <= eps 인 가장 작은 k."""
    return int(np.ceil(np.log(eps * (1 - gamma) / rmax) / np.log(gamma)))
