"""생산성 도구 1장 일반 모형의 NumPy 구현. productivity_ch01_sim.generic_python 과 같은 모형을 한꺼번에 뽑아 센다.

값 하나를 k곳에 두고 m번 바꾼다. 변경마다 나머지 k-1곳이 각각 확률 q로 맞춰진다(correlated면 함께).

무엇을 재는가: 세 가지 비율. (1) m번 바꾸는 동안 한 번도 안 어긋날 확률, (2) 마지막 변경 직후 모두 맞을 확률, (3) 변경당 어긋난 자리 수의 평균.
입력: k(값이 적힌 자리 수), q(고친 사람이 다른 자리 하나를 맞출 확률), m(변경 횟수), n(반복 수), seed. 출력: 위 세 값의 튜플(float).
같은 모형의 식은 productivity_ch01_sim.generic_exact, 순수 파이썬 구현은 generic_python이고 세 값이 오차 안에서 같아야 한다.
"""

import os

# NumPy를 import하기 전에 설정해야 스레드 수 제한이 적용된다(import 뒤에 바꾸면 효과가 없다)
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402


def generic_numpy(k, q, m, n, seed, correlated=False):
    """순수 파이썬 generic_python과 같은 모형을 배열로 한꺼번에 뽑는다. 난수열이 달라 값은 같지 않고 오차 안에서만 일치한다."""
    rng = np.random.default_rng(seed)
    if k == 1:  # 자리가 하나면 맞출 대상이 없어 항상 일치한다
        return 1.0, 1.0, 0.0
    if correlated:
        hit = rng.random((n, m)) < q                       # 변경마다 한 번: 다 맞추거나 다 놓치거나
        miss = np.where(hit, 0, k - 1)
    else:
        # 배열 모양 (n 반복, m 변경, k-1 다른 자리). 난수 >= q이면 그 자리를 놓친 것(맞출 확률이 q)
        miss = (rng.random((n, m, k - 1)) >= q).sum(axis=2)  # 변경마다 놓친 자리 수
    ever = (miss.sum(axis=1) == 0).mean()                 # 한 번도 안 어긋남
    final = (miss[:, -1] == 0).mean()                     # 마지막 변경이 다 닿았나(통째로 바꾸는 값이라 앞의 실수는 덮인다)
    wrong = miss.mean()                                   # 변경당 어긋난 자리
    return float(ever), float(final), float(wrong)
