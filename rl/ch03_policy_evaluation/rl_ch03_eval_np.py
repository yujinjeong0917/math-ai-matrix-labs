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
    # 식: (I - gamma P^pi) V = r^pi 의 좌변 행렬(S x S)과 우변 r^pi(S)를 solve에 그대로 넘긴다
    # 함정: gamma = 1이면 I - P^pi 가 특이행렬이 될 수 있다(test_gamma_one_* 참고). 이 함수는 gamma < 1을 가정
    return np.linalg.solve(np.eye(S) - gamma * P_pi, r_pi)


def evaluate_iterative(P_pi, r_pi, gamma, tol=1e-6, V_true=None, max_sweeps=1_000_000, V0=None):
    """조밀 행렬로 백업을 되풀이한다.

    V_true가 있으면 "정답과의 오차 < tol"에서 멈추고, 없으면 실제 상황처럼
    "앞뒤 스윕의 차이 < tol"에서 멈춘다. 반환: (V, sweeps, errs) errs[k] = k번째 스윕 뒤의 기준값.
    """
    V = np.zeros_like(r_pi) if V0 is None else V0.copy()  # V_0 = 0 이 기본 시작값
    errs = []
    for k in range(1, max_sweeps + 1):
        # 식: V_{k+1} = r^pi + gamma P^pi V_k  (벨만 기대 갱신, 모든 칸을 이전 V로 동시에 계산하는 야코비식)
        V_new = r_pi + gamma * (P_pi @ V)
        # 멈춤 기준: 무한 노름 ||V_new - V_true|| 또는 ||V_new - V|| (칸별 절댓값 차이의 최댓값)
        e = np.max(np.abs(V_new - V_true)) if V_true is not None else np.max(np.abs(V_new - V))
        V = V_new
        errs.append(float(e))
        # 수렴 허용오차: e < tol 이면 종료. V_true가 없을 때 e는 "정답과의 오차"가 아니라 앞뒤 차이다
        if e < tol:
            return V, k, errs
    raise RuntimeError("max_sweeps 안에 멈추지 않았어요")


def evaluate_iterative_sparse(nbr, prob, r_pi, gamma, tol=1e-6, V_true=None, max_sweeps=1_000_000):
    """같은 백업을 칸마다 이웃 4곳만 보고 한다. 스윕당 곱셈이 |S|^2 대신 4|S|."""
    V = np.zeros_like(r_pi)
    errs = []
    for k in range(1, max_sweeps + 1):
        # 식: V_{k+1}(s) = r^pi(s) + gamma sum_a prob[s,a] V_k(nbr[s,a])  (P^pi V 를 이웃 4곳의 합으로 계산)
        # V[nbr]: (S, 4) 이웃 칸의 가치, prob: (S, 4) 그 칸으로 갈 확률. 종료 칸은 자기 자신만 가리킨다
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
    # 칸마다 파이썬 반복을 돌리므로 numpy 원소 접근 대신 리스트로 바꿔 쓴다(속도 목적)
    nb, pr, rr = nbr.tolist(), prob.tolist(), r_pi.tolist()
    v = V.tolist()
    errs = []
    for k in range(1, max_sweeps + 1):
        delta = 0.0  # 이번 스윕에서 한 칸이 바뀐 최대 크기
        for s in order.tolist():
            n4, p4 = nb[s], pr[s]
            # 식: V(s) <- r^pi(s) + gamma sum_a prob[s,a] V(nbr[s,a])  (제자리 기대 갱신)
            # v를 바로 덮어쓰므로 이번 스윕에서 이미 고친 칸은 새 값이 쓰인다(두 배열 야코비식과의 차이)
            new = rr[s] + gamma * (p4[0] * v[n4[0]] + p4[1] * v[n4[1]] + p4[2] * v[n4[2]] + p4[3] * v[n4[3]])
            d = abs(new - v[s])
            if d > delta:
                delta = d
            v[s] = new
        V = np.array(v)
        # V_true가 없으면 스윕 중 최대 변화량 delta로 멈춘다(앞뒤 차이 기준)
        e = np.max(np.abs(V - V_true)) if V_true is not None else delta
        errs.append(float(e))
        # 수렴 허용오차: e < tol
        if e < tol:
            return V, k, errs
    raise RuntimeError("max_sweeps 안에 멈추지 않았어요")


def spectral_radius_nonterminal(P_pi, term):
    """종료 칸을 뺀 P^pi의 스펙트럼 반경. 오차가 실제로 줄어드는 비율은 gamma * 이 값이다."""
    keep = ~term
    # 종료 칸의 행·열을 지운 부분 행렬. 종료 칸은 자기 자신으로만 가서 고유값 1을 만들기 때문에 뺀다
    Q = P_pi[np.ix_(keep, keep)]
    return float(np.max(np.abs(np.linalg.eigvals(Q))))


def predicted_sweeps(eps, gamma, e0=1.0, rate=None):
    """gamma^k * e0 <= eps 를 만족하는 가장 작은 k. rate를 주면 gamma 대신 그 비율로."""
    q = gamma if rate is None else rate
    # 식: q^k e0 <= eps  =>  k >= log(eps/e0) / log(q). 0<q<1 이라 두 로그가 모두 음수이고, 올림해 정수 스윕 수로 만든다
    return int(np.ceil(np.log(eps / e0) / np.log(q)))
