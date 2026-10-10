"""6장 배포 판정: 지금까지 모인 숫자로 '올린다 / 더 본다 / 되돌린다'를 정한다.

decide(metrics_v1, metrics_v2, guard) -> "promote" | "hold" | "rollback"
- metrics_*: {"n": 받은 요청 수, "errors": 틀린 요청 수, "seg": {구간: {"n", "errors"}}}  ("seg"는 없어도 된다)
- guard:
    "z"          되돌릴 문턱(한쪽 검정 z)
    "min_n"      판단을 시작할 최소 v2 요청 수
    "promote_n"  올리기 전에 v2가 받아야 할 요청 수
    "segments"   True면 구간(여기서는 지역)마다 따로도 비교한다(구간 가드)
    "min_seg_n"  구간 비교를 시작할 그 구간의 최소 v2 요청 수

규칙(교육용 단순화)
1. v2 요청이 min_n보다 적으면 hold. 표본이 너무 작으면 우연히 튀는 오류율에 끌려다닌다.
2. 전체 오류율 차이를 표준오차로 나눈 z('v2가 더 나쁘다' 방향)가 문턱을 넘으면 rollback.
3. 구간 가드가 켜져 있으면 구간마다 같은 검사를 해서 하나라도 넘으면 rollback.
4. 아무것도 넘지 않았고 v2 요청이 promote_n 이상이면 promote. 아니면 hold.
여러 번 들여다보거나 여러 구간을 볼 때의 오탐 보정(순차검정, 다중비교 보정)은 일부러 하지 않는다.
그 대가는 experiments.py의 E4에서 잰다.
"""

import math


def z_worse(m1, m2):
    """v2 오류율이 v1보다 높은 방향의 z. 합동 비율로 표준오차를 구한다."""
    n1, n2 = m1["n"], m2["n"]
    if n1 == 0 or n2 == 0:
        return 0.0
    e1, e2 = m1["errors"] / n1, m2["errors"] / n2
    pool = (m1["errors"] + m2["errors"]) / (n1 + n2)
    se = math.sqrt(pool * (1.0 - pool) * (1.0 / n1 + 1.0 / n2))
    if se == 0.0:
        return 0.0 if e2 <= e1 else math.inf
    return (e2 - e1) / se


def worst_segment(m1, m2, min_seg_n):
    """(구간, z) 중 z가 가장 큰 것. 비교할 수 있는 구간이 없으면 (None, 0.0)."""
    best = (None, 0.0)
    for seg, s2 in m2.get("seg", {}).items():
        s1 = m1.get("seg", {}).get(seg)
        if s1 is None or s2["n"] < min_seg_n or s1["n"] < min_seg_n:
            continue
        z = z_worse(s1, s2)
        if z > best[1]:
            best = (seg, z)
    return best


def decide(metrics_v1, metrics_v2, guard):
    if metrics_v2["n"] < guard["min_n"]:
        return "hold"
    if z_worse(metrics_v1, metrics_v2) > guard["z"]:
        return "rollback"
    if guard.get("segments"):
        _, z = worst_segment(metrics_v1, metrics_v2, guard.get("min_seg_n", 20))
        if z > guard["z"]:
            return "rollback"
    if metrics_v2["n"] >= guard["promote_n"]:
        return "promote"
    return "hold"


DEFAULT_GUARD = {"z": 3.0, "min_n": 50, "promote_n": 400, "segments": True, "min_seg_n": 20}
OVERALL_ONLY = dict(DEFAULT_GUARD, segments=False)
