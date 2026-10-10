"""6장 배포 시뮬레이터: 같은 요청 흐름에 배포 방식만 바꿔 대 본다.

요청 i마다 v1과 v2가 각각 맞혔는지(c1[i], c2[i])를 먼저 다 계산해 둔다. 배포 방식은 '그 요청을 누가 받았나'만 정한다.
그래서 피해는 추정이 아니라 같은 요청에서의 정확한 반사실 비교다.
  피해(늘어난 오류) = v2가 받은 요청에서 (v2가 틀림) - (v1이 틀림) 의 합 = sum(c1 - c2)

가정(교육용 단순화)
- 정답(이탈했는지)은 label_delay 요청 뒤에 들어온다. 0이면 요청 직후. 실제 이탈 라벨은 몇 주 뒤에 온다.
- 감시는 check_every 요청마다 한 번 한다. 그때까지 정답이 들어온 요청만 센다.
- 전체 교체(full): 처음부터 모든 요청을 v2로. 비교 대상은 배포 전 v1 구간의 숫자(baseline, 전후 비교).
  되돌리기로 정해도 v1을 다시 배포하는 동안 rollback_delay 요청은 계속 v2가 받는다.
- 블루그린(bluegreen): 전체 교체와 같지만 v1 환경이 그대로 켜져 있어 라우터만 돌리면 된다(지연 0).
- 카나리(canary): 해시로 고른 비율 p의 사용자만 v2. 비교 대상은 같은 시간대의 v1 요청(동시 대조군).
  promote가 나오면 그다음 요청부터 모두 v2로 보내고 감시를 멈춘다. rollback이면 곧바로 모두 v1.
"""

import numpy as np

from pipelines_ch06_release_gate import decide
from pipelines_ch06_router import route


def empty_metrics():
    return {"n": 0, "errors": 0, "seg": {}}


def add(m, err, seg):
    m["n"] += 1
    m["errors"] += err
    s = m["seg"].setdefault(int(seg), {"n": 0, "errors": 0})
    s["n"] += 1
    s["errors"] += err


def metrics_of(c, segments):
    """맞힘 배열에서 감시용 숫자를 만든다(배포 전 v1 구간의 기준값용)."""
    m = empty_metrics()
    for ci, sg in zip(np.asarray(c), np.asarray(segments)):
        add(m, int(1 - ci), sg)
    return m


def simulate_traffic(c1, c2, user_ids, segments, kind, guard, p=1.0, baseline=None, rollback_delay=0,
                     check_every=100, label_delay=0, single_look_at=None):
    """segments: 요청마다 구간(지역 코드).
    single_look_at: 주면 감시를 정답이 들어온 v2 요청 수가 그 값에 닿을 때 한 번만 한다(오탐 비교용).
    """
    n = len(c1)
    c1 = np.asarray(c1)
    c2 = np.asarray(c2)
    segments = np.asarray(segments)
    if kind == "canary":
        in_canary = np.array([route(u, p) == "v2" for u in user_ids])
    else:
        in_canary = np.ones(n, dtype=bool)
    served_v2 = np.zeros(n, dtype=bool)
    watched = np.zeros(n, dtype=bool)  # 감시 단계(release)에서 받은 요청
    m1 = empty_metrics()
    m2 = empty_metrics()
    state = "release"  # release -> promoted | rolling_back -> rolled_back
    decided_at = promoted_at = None
    cut_at = None  # 이 요청부터 v1
    looked = False
    j = 0  # 정답이 들어와 센 요청의 끝
    for i in range(n):
        if state == "promoted":
            v2 = True
        elif state == "rolled_back":
            v2 = False
        elif state == "rolling_back":
            v2 = i < cut_at
            if not v2:
                state = "rolled_back"
        else:
            v2 = bool(in_canary[i])
        served_v2[i] = v2
        watched[i] = state == "release"
        if state != "release":
            continue
        while j <= i - label_delay:
            if watched[j]:
                if served_v2[j]:
                    add(m2, int(1 - c2[j]), segments[j])
                elif kind == "canary":
                    add(m1, int(1 - c1[j]), segments[j])
            j += 1
        if single_look_at is not None:
            if looked or m2["n"] < single_look_at:
                continue
            looked = True
        elif (i + 1) % check_every != 0:
            continue
        ref = m1 if kind == "canary" else baseline
        d = decide(ref, m2, guard)
        if single_look_at is not None and d == "hold":
            d = "promote"
        if d == "rollback":
            decided_at = i + 1
            delay = rollback_delay if kind == "full" else 0
            cut_at = i + 1 + delay
            state = "rolling_back" if delay > 0 else "rolled_back"
        elif d == "promote" and kind == "canary":
            promoted_at = i + 1
            state = "promoted"
    harm = int(np.sum((c1 - c2)[served_v2]))
    return {
        "decision": "rollback" if decided_at is not None else ("promote" if promoted_at is not None else "none"),
        "decided_at": decided_at,
        "promoted_at": promoted_at,
        "served_v2": int(served_v2.sum()),
        "v2_wrong": int(np.sum(1 - c2[served_v2])),
        "harm": harm,
        "harm_by_segment": {int(g): int(np.sum((c1 - c2)[served_v2 & (segments == g)])) for g in np.unique(segments)},
        "v2_labeled_at_decision": m2["n"],
    }
