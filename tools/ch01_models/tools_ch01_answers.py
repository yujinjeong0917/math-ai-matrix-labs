"""AI 도구 실전 1장: 같은 질문에 답이 갈리는 이유를 숫자로 흉내 낸다.

- 모델은 다음 말 후보마다 점수를 매기고, 점수를 확률로 바꿔 하나를 뽑는다.
- 확률 = e^(점수 / 온도T) 를 모두 더한 값으로 나눈 것(온도를 넣은 소프트맥스).
- 온도를 낮추면 1등 후보 쪽으로 쏠리고, 온도 0에 가까우면 늘 1등만 고른다(탐욕 선택).
- 다른 모델은 같은 질문에 다른 점수표를 갖는다(학습 데이터·사후 학습이 다르므로). 점수표는 직접 정한 장난감 값이다.

표준 라이브러리만 쓴다(math, random). 시드를 고정해 결과가 매번 같다.
"""

import math
import random

QUESTION = "이번 주 고객 문의에서 가장 먼저 다룰 유형은"
MODEL_A = {"배송": 2.0, "환불": 1.0, "결제 오류": 0.0}
MODEL_B = {"배송": 1.0, "환불": 2.0, "결제 오류": 0.5}


def softmax(scores, T=1.0):
    """점수 dict를 온도 T의 소프트맥스로 확률 dict로 바꾼다. 확률의 합은 1."""
    if T <= 0:
        raise ValueError("온도는 0보다 커야 해요. 0에 가까운 경우는 greedy()를 쓰세요.")
    # 수치 안정화: 최댓값을 빼도 분자·분모에서 e^(-mx/T)가 약분돼 확률이 같고, exp 폭주(overflow)를 막는다.
    mx = max(scores.values())
    # 식: e^((점수-mx)/T)
    ex = {k: math.exp((v - mx) / T) for k, v in scores.items()}
    s = sum(ex.values())
    return {k: v / s for k, v in ex.items()}


def greedy(scores):
    """점수가 가장 높은 후보 하나(온도 0 극한). 동점이면 먼저 나온 키."""
    return max(scores, key=scores.get)


def sample(scores, T, n, seed):
    """확률대로 n번 뽑는다. 반환: (후보별 횟수 dict, 뽑힌 순서 리스트)."""
    # 전역 난수 상태를 건드리지 않는 독립 난수기. 시드를 고정해야 실행마다 같은 횟수가 나온다.
    rng = random.Random(seed)
    p = softmax(scores, T)
    keys = list(p)
    out = rng.choices(keys, weights=[p[k] for k in keys], k=n)
    return {k: out.count(k) for k in keys}, out


def hand_ratios(scores, T):
    """손계산용: e^(점수/T) 값 그대로(최댓값 빼기 없이)."""
    return {k: math.exp(v / T) for k, v in scores.items()}
