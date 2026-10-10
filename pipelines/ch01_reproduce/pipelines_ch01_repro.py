"""1장 재현 도구: 시드 고정, 환경 기록, 부동소수 합의 순서 실험.

env_snapshot은 결과 파일에 그대로 실려 공개되므로 컴퓨터 이름, 사용자 이름, 홈 경로는 넣지 않는다.
"""

import math
import os
import platform
import random
import sys

import numpy as np

ENV_KEYS = ["python", "numpy", "torch", "os", "machine", "processor", "cpu_count", "omp_num_threads",
            "torch_num_threads", "byteorder"]


def set_seeds(seed):
    """전역 난수 생성기 세 개(파이썬, NumPy 전역, PyTorch)를 한꺼번에 고정한다.

    이 저장소의 학습 코드는 전역 생성기 대신 default_rng(seed)를 직접 넘기지만,
    라이브러리 안쪽에서 전역 생성기를 쓰는 경우를 막으려고 같이 고정한다.
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def env_snapshot():
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch_num_threads": torch.get_num_threads(),
        "byteorder": sys.byteorder,
    }


def env_diff(a, b):
    """두 환경 기록에서 다른 항목만."""
    return {k: (a.get(k), b.get(k)) for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}


def sequential_sum(x):
    """앞에서부터 한 개씩 더한다(가장 단순한 순서). 결과 자료형은 x와 같다."""
    acc = x.dtype.type(0)
    for v in x:
        acc = acc + v
    return acc


def chunked_sum(x, k):
    """k개 일꾼이 나눠 더한 뒤 합치는 병렬 합을 흉내 낸다. k가 바뀌면 더하는 순서가 바뀐다."""
    parts = [np.sum(c) for c in np.array_split(x, k)]
    acc = x.dtype.type(0)
    for p in parts:
        acc = acc + p
    return acc


def exact_sum(x):
    """math.fsum: 반올림 오차 없이 더한 값(비교 기준)."""
    return math.fsum(float(v) for v in x)
