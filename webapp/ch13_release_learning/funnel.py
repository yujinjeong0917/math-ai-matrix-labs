"""단계 전환율로 먼저 조사할 단계 찾기 (1인 개발 앱 출시 실전 13장).

입력: [(단계 이름, 사용자 수), ...] 순서대로, 줄어들기만 해야 한다.
출력: {"rows": [{"stage", "count", "step_rate"}], "investigate_first": 직전 대비 전환율이 가장 낮은 단계 이름}
"""

def analyze(counts):
    # 퍼널은 단계가 뒤로 갈수록 줄어드는 것이 전제라 늘어나면 데이터 오류로 본다
    if not counts or any(v < 0 for _, v in counts) or any(counts[i][1] < counts[i + 1][1] for i in range(len(counts) - 1)):
        raise ValueError("counts must be non-negative and non-increasing")
    rows, worst = [], None
    for i, (stage, count) in enumerate(counts):
        # 단계 전환율 = 이번 단계 수 / 직전 단계 수. 첫 단계나 직전이 0이면 정의 안 됨(None)
        rate = None if i == 0 or counts[i - 1][1] == 0 else count / counts[i - 1][1]
        row = {"stage": stage, "count": count, "step_rate": rate}
        rows.append(row)
        # 전환율이 가장 낮은 단계 = 가장 크게 이탈한 단계(처음 나온 것을 유지: < 이므로 동률이면 앞 단계)
        if rate is not None and (worst is None or rate < worst["step_rate"]):
            worst = row
    return {"rows": rows, "investigate_first": None if worst is None else worst["stage"]}
