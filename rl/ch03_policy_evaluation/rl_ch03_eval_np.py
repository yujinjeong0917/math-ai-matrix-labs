"""강화학습 3장 NumPy 최소 구현: 고정 정책의 가치를 구하는 세 가지 방법.

정책을 고정하면 벨만 기대방정식은 미지수가 상태 수만큼인 연립일차방정식이다.
    V = r^pi + gamma P^pi V   <=>   (I - gamma P^pi) V = r^pi

1) 직접 풀이   V = solve(I - gamma P^pi, r^pi)               LU 분해, 약 (2/3)|S|^3 번의 곱셈·덧셈
2) 반복 평가   V_{k+1} = r^pi + gamma P^pi V_k   (야코비식)  조밀하면 스윕당 2|S|^2, 희소하면 2|S|*4
3) 제자리 반복 V(s) <- r^pi(s) + gamma sum_s' P^pi(s,s') V(s')를 칸마다 바로 덮어씀 (가우스-자이델식)

오차 보장: ||V_k - V^pi||_inf <= gamma^k ||V_0 - V^pi||_inf
"""

import numpy as np


def evaluate_direct(P_pi, r_pi, gamma):
    """(I - gamma P^pi) V = r^pi 를 LAPACK gesv(LU 분해)로 한 번에 푼다."""
    S = P_pi.shape[0]
    return np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)


def evaluate_iterative(P_pi, r_pi, gamma, tol=1e-6, V_true=None, max_sweeps=1_000_000, V0=None):
    """조밀 행렬로 백업을 되풀이한다.

    V_true가 있으면 "정답과의 오차 < tol"에서 멈추고, 없으면 실제 상황처럼
    "앞뒤 스윕의 차이 < tol"에서 멈춘다. 반환: (V, sweeps, errs) errs[k] = k번째 스윕 뒤의 기준값.
    """
    V = np.zeros_like(r_pi) if V0 is None else V0.copy()
    errs = []
    for k in range(1, max_sweeps + 1):
        V_new = r_pi + gamma * (P_pi @ V)
        e = np.max(np.abs(V_new - V_true)) if V_true is not None else np.max(np.abs(V_new - V))
        V = V_new
        errs.append(float(e))
        if e < tol:
            return V, k, errs
    raise RuntimeError("max_sweeps 안에 멈추지 않았어요")


def evaluate_iterative_sparse(nbr, prob, r_pi, gamma, tol=1e-6, V_true=None, max_sweeps=1_000_000):
    """같은 백업을 칸마다 이웃 4곳만 보고 한다. 스윕당 곱셈이 |S|^2 대신 4|S|."""
    V = np.zeros_like(r_pi)
    errs = []
    for k in range(1, max_sweeps + 1):
        V_new = r_pi + gamma * (prob * V[nbr]).sum(axis=1)
        e = np.max(np.abs(V_new - V_true)) if V_true is not None else np.max(np.abs(V_new - V))
        V = V_new
        errs.append(float(e))
        if e < tol:
            return V, k, errs
    raise RuntimeError("max_sweeps 안에 멈추지 않았어요")


def evaluate_inplace(nbr, prob, r_pi, gamma, tol=1e-6, V_true=None, order=None, max_sweeps=1_000_000):
    """제자리 갱신(가우스-자이델식). 한 스윕 안에서 먼저 고친 칸의 새 값을 바로 쓴다.

    order: 칸을 고치는 순서(기본은 0, 1, 2, ...). 순서가 수렴 속도를 바꾼다.
    """
    S = r_pi.shape[0]
    order = np.arange(S) if order is None else np.asarray(order)
    V = np.zeros(S)
    nb, pr, rr = nbr.tolist(), prob.tolist(), r_pi.tolist()
    v = V.tolist()
    errs = []
    for k in range(1, max_sweeps + 1):
        delta = 0.0
        for s in order.tolist():
            n4, p4 = nb[s], pr[s]
            new = rr[s] + gamma * (p4[0] * v[n4[0]] + p4[1] * v[n4[1]] + p4[2] * v[n4[2]] + p4[3] * v[n4[3]])
            d = abs(new - v[s])
            if d > delta:
                delta = d
            v[s] = new
        V = np.array(v)
        e = np.max(np.abs(V - V_true)) if V_true is not None else delta
        errs.append(float(e))
        if e < tol:
            return V, k, errs
    raise RuntimeError("max_sweeps 안에 멈추지 않았어요")


def spectral_radius_nonterminal(P_pi, term):
    """종료 칸을 뺀 P^pi의 스펙트럼 반경. 오차가 실제로 줄어드는 비율은 gamma * 이 값이다."""
    keep = ~term
    Q = P_pi[np.ix_(keep, keep)]
    return float(np.max(np.abs(np.linalg.eigvals(Q))))


def predicted_sweeps(eps, gamma, e0=1.0, rate=None):
    """gamma^k * e0 <= eps 를 만족하는 가장 작은 k. rate를 주면 gamma 대신 그 비율로."""
    q = gamma if rate is None else rate
    return int(np.ceil(np.log(eps / e0) / np.log(q)))
