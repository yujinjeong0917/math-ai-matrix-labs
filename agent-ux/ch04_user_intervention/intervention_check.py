"""4장 실습: He 외(2025)의 보고값을 다시 세고, "보정된 신뢰" 점수가 계획 구성에 따라 움직이는 것을 가상 데이터로 본다.

논문 값은 논문이 보고한 값이다. 가상이라고 적은 계획 묶음은 이 장의 예시용이고 사람이 어떻게 믿는지를 재현하지 않는다.
이 코드는 점수가 어떻게 계산되는지 보여 주는 용도다. 표준 라이브러리만 쓴다.
"""

# He 외(2025) 보고값
CELLS = {"AP-AE": 63, "AP-UE": 64, "UP-AE": 61, "UP-UE": 60}  # 계획 자동/개입 x 실행 자동/개입
RECRUITED = 347
PLAN_EDITED_OF_UP = (104, 121)    # 계획 개입 참가자 중 계획을 한 번 이상 편집한 사람
EXEC_CHANGED_OF_UE = (114, 124)   # 실행 개입 참가자 중 행동 예측을 바꾸는 상호작용을 한 사람
PLAN_BUTTON_COUNTS = {"삭제": 394, "추가": 183, "분할": 126}
EXEC_COUNTS = {"행동 직접 지정": 445, "실행 전 피드백": 91, "실행 후 피드백": 163}
TASKS_Q5 = (717, 235)  # 계획 품질 5점 과제 수, 그중 틀린 실행 수
REPORTED_Q5_ACC = 66.7
FAIL_CAUSE = {"매개변수 오예측": 48.9, "무효 행동": 48.5, "행동 이름 오예측": 2.6}
ACC_BY_QUALITY = {"1~2점": 1.8, "3~4점": 59.0, "5점": 66.7}
HE_META = {
    "에이전트": "GPT-3.5-turbo",
    "계획 품질 평균(자동, 개입)": (3.8, (3.6, 3.7)),
    "계획 품질이 떨어진 과제를 걸러냈다는 서술 위치": "5.2.4절",
}
CTP_RANGE = {"위험 높은 과제 1~3": (0.03, 0.27), "위험 낮은 과제 4~6": (0.75, 0.97)}

# 가상 계획 묶음: 전문가 품질 점수(5점이 고품질)
POOLS = {
    "고품질이 많은 묶음": [5, 5, 5, 5, 5, 4],
    "결함이 많은 묶음": [2, 3, 4, 5, 2, 3],
}


def pct(x, n):
    return round(100 * x / n, 1)


def calibrated_trust(qualities, trusted):
    """결정마다 '5점이면 믿고 5점 미만이면 믿지 않기'를 맞힌 비율."""
    ok = sum(1 for q, t in zip(qualities, trusted) if (q == 5) == t)
    return ok / len(qualities)


def always_trust(qualities):
    return calibrated_trust(qualities, [True] * len(qualities))


def always_distrust(qualities):
    return calibrated_trust(qualities, [False] * len(qualities))


def ideal_reviewer(qualities):
    return calibrated_trust(qualities, [q == 5 for q in qualities])


def main():
    total = sum(CELLS.values())
    print("[1] 칸 인원 합 %d명(모집 %d명), 계획 개입 %d명, 실행 개입 %d명" % (
        total, RECRUITED, CELLS["UP-AE"] + CELLS["UP-UE"], CELLS["AP-UE"] + CELLS["UP-UE"]))
    print("[2] 계획을 한 번 이상 편집 %d/%d = %.1f%%, 행동 예측을 바꾼 상호작용 %d/%d = %.1f%%" % (
        PLAN_EDITED_OF_UP[0], PLAN_EDITED_OF_UP[1], pct(*PLAN_EDITED_OF_UP),
        EXEC_CHANGED_OF_UE[0], EXEC_CHANGED_OF_UE[1], pct(*EXEC_CHANGED_OF_UE)))
    n, wrong = TASKS_Q5
    print("[3] 계획 품질 5점 과제 %d개 중 %d개 오류: (%d-%d)/%d = %.1f%% (논문 보고 %.1f%%)" % (
        n, wrong, n, wrong, n, pct(n - wrong, n), REPORTED_Q5_ACC))
    print("[4] 오류 원인 비율 합 %.1f%%" % sum(FAIL_CAUSE.values()))
    print("[5] 가상 계획 묶음의 보정 점수(CT)")
    for name, q in POOLS.items():
        good = sum(1 for x in q if x == 5)
        print("  %s %s: 고품질 %d/%d, 항상 믿기 %.3f, 항상 믿지 않기 %.3f, 완벽한 검토자 %.3f" % (
            name, q, good, len(q), always_trust(q), always_distrust(q), ideal_reviewer(q)))


if __name__ == "__main__":
    main()
