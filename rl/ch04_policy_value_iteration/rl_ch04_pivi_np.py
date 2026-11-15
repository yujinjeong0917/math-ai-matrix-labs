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

TIE_TOL = 1e-9  # 동률 판정 허용오차: Q가 최댓값보다 1e-9 이내로만 작으면 같은 값으로 본다(부동소수 잡음 흡수)


def q_from_v(V, P, R, gamma):
    """Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')]. P, R: (S, A, S), V: (S,), 반환 Q: (S, A)."""
    # 식: Q(s,a) = sum_s' P(s'|s,a) [R(s,a,s') + gamma V(s')].  V[None, None, :]은 (1, 1, S)로 모든 (s, a)에 s'별 V를 붙임
    return np.einsum("sap,sap->sa", P, R + gamma * V[None, None, :])


def bellman_T(V, P, R, gamma):
    """벨만 최적 연산자 T. 가치 반복의 한 스윕."""
    # 식: (T V)(s) = max_a Q(s,a)  (벨만 최적 방정식의 우변). V* 는 T V* = V* 인 고정점
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
    # 식: pi'(s) = argmax_a Q(s,a). 단, 최댓값에서 tol 이내인 행동은 모두 동률 후보(best[S, A] 불리언)로 둔다
    best = Q >= Q.max(axis=1, keepdims=True) - tol
    pi = best.argmax(axis=1)  # 불리언 배열의 argmax는 첫 번째 True의 위치(번호가 가장 작은 후보)
    if tie == "random":
        # 후보가 둘 이상인 칸만 무작위로 다시 뽑는다
        for s in np.flatnonzero(best.sum(axis=1) > 1):
            pi[s] = rng.choice(np.flatnonzero(best[s]))
    elif tie == "keep" and old is not None:
        # 현재 행동이 후보 안에 있으면 바꾸지 않는다. 정책 개선 정리의 "엄격히 좋아질 때만 바꾼다"는 방향
        keep = best[np.arange(len(old)), old]
        pi[keep] = old[keep]
    return pi


def policy_mats(pi, P, R):
    """결정론적 정책 pi[S]에서 P^pi[S,S], r^pi[S]."""
    idx = np.arange(P.shape[0])
    Ppi = P[idx, pi]  # 칸 s마다 행동 pi[s]의 행을 뽑는다: (S, S). P^pi[s,s'] = P(s'|s, pi(s))
    # 식: r^pi(s) = sum_s' P(s'|s,pi(s)) R(s,pi(s),s')
    rpi = (Ppi * R[idx, pi]).sum(axis=1)
    return Ppi, rpi


def evaluate_exact(pi, P, R, gamma):
    """3장의 직접 풀이: (I - gamma P^pi) V = r^pi."""
    Ppi, rpi = policy_mats(pi, P, R)
    # 식: (I - gamma P^pi) V^pi = r^pi  (정책 평가, 벨만 기대방정식의 선형계 형태)
    return np.linalg.solve(np.eye(P.shape[0]) - gamma * Ppi, rpi)


def evaluate_sweeps(pi, P, R, gamma, V, k):
    """3장의 반복 평가를 k스윕만(두 배열, 야코비식). 백업 수 = k*S."""
    Ppi, rpi = policy_mats(pi, P, R)
    for _ in range(k):
        # 식: V <- r^pi + gamma P^pi V  (벨만 기대 갱신, 매 스윕 모든 칸을 이전 V로 동시에)
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
        V = evaluate_exact(pi, P, R, gamma)  # 평가: 식 (I - gamma P^pi) V = r^pi 를 정확히 푼다
        new = greedy_policy(V, P, R, gamma, tie=tie, rng=rng, old=pi)  # 개선: 식 pi'(s) = argmax_a Q^pi(s,a)
        changed = int((new != pi).sum())
        history.append({"changed": changed, "V": V})  # V는 개선 전 정책 pi의 가치
        pi = new
        # 종료 조건: 정책이 한 칸도 안 바뀌면 최적(tie="random"이면 동률 칸이 계속 뒤집혀 이 조건이 안 걸릴 수 있다)
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
        # 식: V_{k+1}(s) = max_a sum_s' P(s'|s,a) [R(s,a,s') + gamma V_k(s')]  (벨만 최적 갱신)
        V_new = bellman_T(V, P, R, gamma)
        d = float(np.max(np.abs(V_new - V)))  # 앞뒤 스윕 차이(무한 노름)
        e = None if V_star is None else float(np.max(np.abs(V_new - V_star)))  # V*를 알 때만 실제 오차
        history.append((d, e))
        V = V_new
        # 수렴 허용오차: d < theta. 정답 오차가 아니라 앞뒤 차이 기준이라 실제 오차는 최대 gamma/(1-gamma) * theta 까지 남을 수 있다
        # 주의: max_iter를 다 써도 예외 없이 마지막 V를 반환한다(수렴 여부는 history로 확인)
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
        # 개선 스윕: 식 TV(s) = max_a Q(s,a), pi(s) = argmax_a Q(s,a). 동률 판정 없이 argmax 그대로(최소 번호 행동)
        TV, pi = Q.max(axis=1), Q.argmax(axis=1)
        backups += S * A  # 모든 (s, a)의 기댓값을 한 번씩
        d = float(np.max(np.abs(TV - V)))  # 식: ||T V - V||_inf
        # 평가 스윕 k번: 식 V <- r^pi + gamma P^pi V, 시작값은 개선 결과 TV. k=0이면 V = TV (가치 반복)
        V = evaluate_sweeps(pi, P, R, gamma, TV, k)
        backups += k * S  # 평가 한 스윕은 칸당 행동 하나(pi)만 보므로 S
        # 수렴 허용오차: d < theta (평가 스윕을 마친 뒤에 검사하므로 k > 0이면 반환 V는 TV에 평가 스윕 k번을 더한 값)
        if d < theta:
            return pi, V, it, backups
    raise RuntimeError("max_iter 안에 멈추지 않았어요")


def theory_vi_sweeps(eps, gamma, rmax=1.0):
    """V0=0에서 ||V0 - V*|| <= rmax/(1-gamma) 이므로 gamma^k rmax/(1-gamma) <= eps 인 가장 작은 k."""
    # 식: gamma^k rmax/(1-gamma) <= eps  =>  k >= log(eps (1-gamma)/rmax) / log(gamma). 올림해 정수 스윕 수로 만든다
    # 가정: 보상 절댓값이 rmax 이하라서 ||V*|| <= rmax/(1-gamma) 이고 V0 = 0 (최악 경우의 상한이라 실제 스윕 수는 더 적을 수 있다)
    return int(np.ceil(np.log(eps * (1 - gamma) / rmax) / np.log(gamma)))
