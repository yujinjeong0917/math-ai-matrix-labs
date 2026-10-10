"""9장 배포 판정: 6장 pipelines_ch06_release_gate.py의 z_worse()와 decide()를 그대로 복사했다(구간 가드 부분은 뺐다).

decide(metrics_old, metrics_new, guard) -> "promote" | "hold" | "rollback"
- metrics_*: {"n": 요청 수, "errors": 틀린 요청 수}
1. 새 모델 요청이 min_n보다 적으면 hold.
2. 오류율 차이를 표준오차로 나눈 z('새 모델이 더 나쁘다' 방향)가 문턱을 넘으면 rollback.
3. 넘지 않았고 요청이 promote_n 이상이면 promote, 아니면 hold.
이 판정은 '맞혔는지'를 알아야 하므로 정답(이탈 여부)이 도착한 요청만 셀 수 있다. 9장에서 중요한 점이다.
"""

import math


def z_worse(m1, m2):
    n1, n2 = m1["n"], m2["n"]
    if n1 == 0 or n2 == 0:
        return 0.0
    e1, e2 = m1["errors"] / n1, m2["errors"] / n2
    pool = (m1["errors"] + m2["errors"]) / (n1 + n2)
    se = math.sqrt(pool * (1.0 - pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 0.0 if e2 <= e1 else math.inf
    return (e2 - e1) / se


def decide(metrics_old, metrics_new, guard):
    if metrics_new["n"] < guard["min_n"]:
        return "hold"
    if z_worse(metrics_old, metrics_new) > guard["z"]:
        return "rollback"
    if metrics_new["n"] >= guard["promote_n"]:
        return "promote"
    return "hold"


GUARD = {"z": 3.0, "min_n": 50, "promote_n": 400}
