"""생산성 도구 1장 일반 모형의 NumPy 구현. productivity_ch01_sim.generic_python 과 같은 모형을 한꺼번에 뽑아 센다.

값 하나를 k곳에 두고 m번 바꾼다. 변경마다 나머지 k-1곳이 각각 확률 q로 맞춰진다(correlated면 함께).
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402


def generic_numpy(k, q, m, n, seed, correlated=False):
    rng = np.random.default_rng(seed)
    if k == 1:
        return 1.0, 1.0, 0.0
    if correlated:
        hit = rng.random((n, m)) < q                       # 변경마다 한 번: 다 맞추거나 다 놓치거나
        miss = np.where(hit, 0, k - 1)
    else:
        miss = (rng.random((n, m, k - 1)) >= q).sum(axis=2)  # 변경마다 놓친 자리 수
    ever = (miss.sum(axis=1) == 0).mean()                 # 한 번도 안 어긋남
    final = (miss[:, -1] == 0).mean()                     # 마지막 변경이 다 닿았나(통째로 바꾸는 값이라 앞의 실수는 덮인다)
    wrong = miss.mean()                                   # 변경당 어긋난 자리
    return float(ever), float(final), float(wrong)
