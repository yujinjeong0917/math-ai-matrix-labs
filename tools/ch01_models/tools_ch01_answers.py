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
    if T <= 0:
        raise ValueError("온도는 0보다 커야 해요. 0에 가까운 경우는 greedy()를 쓰세요.")
    mx = max(scores.values())
    ex = {k: math.exp((v - mx) / T) for k, v in scores.items()}
    s = sum(ex.values())
    return {k: v / s for k, v in ex.items()}


def greedy(scores):
    return max(scores, key=scores.get)


def sample(scores, T, n, seed):
    rng = random.Random(seed)
    p = softmax(scores, T)
    keys = list(p)
    out = rng.choices(keys, weights=[p[k] for k in keys], k=n)
    return {k: out.count(k) for k in keys}, out


def hand_ratios(scores, T):
    """손계산용: e^(점수/T) 값 그대로(최댓값 빼기 없이)."""
    return {k: math.exp(v / T) for k, v in scores.items()}
