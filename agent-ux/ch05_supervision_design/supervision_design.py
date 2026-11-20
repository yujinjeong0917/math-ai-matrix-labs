"""5장 실습: 에이전트를 지켜보는 화면을 점검하고 비교하는 세 가지 계산.

1) 위험과 정답 여부를 교차표로 세어 두 요인이 섞였는지 본다(He 외 2025의 과제 구성 서술을 옮긴 것과 균형 설계).
2) 감독 기록(가상)에서 놓치지 않고 잡은 비율, 괜히 멈춘 비율, 판단이 맞은 비율을 센다.
3) 화면 점검표 네 항목을 확인한다. 한계값은 팀이 정하는 값이고 논문 근거가 없다.
모든 기록과 화면 값은 가상, 예시용이다. 표준 라이브러리만 쓴다.
"""
from collections import Counter

# He 외(2025)의 과제 구성: 위험이 높은 과제 1~3은 모두 불완전한 계획, 낮은 과제 4~6은 모두 정확한 계획이라는 서술
HE_TASKS = [("높음", False)] * 3 + [("낮음", True)] * 3
# 비교 설계에 쓴 논문 보고값(검산 대상이 아니라 기록용)
REPORTED = {
    "He 외 표본 계산": {"효과 크기 f": 0.25, "유의수준": 0.0125, "검정력": 0.8, "필요 인원": 244, "분석 인원": 248},
    "Yu 외 참가자 수(집단 내)": 40,
}
RISKS = ("낮음", "높음")
CORRECT = (True, False)


def crosstab(tasks):
    c = Counter(tasks)
    return {(r, ok): c.get((r, ok), 0) for r in RISKS for ok in CORRECT}


def empty_cells(tasks):
    return [k for k, v in crosstab(tasks).items() if v == 0]


def is_balanced(tasks):
    return len(set(crosstab(tasks).values())) == 1 and 0 not in crosstab(tasks).values()


def balanced_design(per_cell):
    return [(r, ok) for r in RISKS for ok in CORRECT for _ in range(per_cell)]


# 가상 감독 기록: 칸(위험, 에이전트가 맞았나)마다 (승인, 수정, 중단) 횟수. 칸마다 6단계, 합 24단계.
LOG_SPEC = {
    "화면 P": {
        ("낮음", True): (5, 1, 0), ("낮음", False): (3, 2, 1),
        ("높음", True): (4, 1, 1), ("높음", False): (1, 3, 2),
    },
    "화면 Q": {
        ("낮음", True): (6, 0, 0), ("낮음", False): (5, 1, 0),
        ("높음", True): (6, 0, 0), ("높음", False): (4, 1, 1),
    },
}


def expand(spec):
    rows = []
    for (risk, ok), (a, e, s) in spec.items():
        rows += [(risk, ok, "승인")] * a + [(risk, ok, "수정")] * e + [(risk, ok, "중단")] * s
    return rows


def metrics(rows, risk=None):
    rows = [r for r in rows if risk is None or r[0] == risk]
    wrong = [r for r in rows if not r[1]]
    right = [r for r in rows if r[1]]
    caught = sum(1 for r in wrong if r[2] != "승인")
    false_alarm = sum(1 for r in right if r[2] != "승인")
    right_calls = caught + sum(1 for r in right if r[2] == "승인")
    return {
        "steps": len(rows), "wrong": len(wrong), "caught": caught, "missed": len(wrong) - caught,
        "false_alarm": false_alarm, "right": len(right), "right_calls": right_calls,
        "catch_rate": round(100 * caught / len(wrong), 1) if wrong else None,
        "false_alarm_rate": round(100 * false_alarm / len(right), 1) if right else None,
        "decision_accuracy": round(100 * right_calls / len(rows), 1),
    }


# 점검표. 한계값은 팀이 정하는 값이고 논문 근거가 없다(가상 5초).
TEAM_GAP_LIMIT_S = 5


def checklist(spec, limit=TEAM_GAP_LIMIT_S):
    return {
        "가장 긴 조용한 구간이 팀 한계 이하": spec["longest_gap_s"] <= limit,
        "긴 단계 중 멈추기 버튼이 있음": spec["stop_button"],
        "되돌릴 수 없는 단계 앞에 확인이 있음": spec["confirm_before_irreversible"],
        "끝에 한 일 목록이 있음": spec["summary_lists_actions"],
    }


SCREENS = {
    "초안(중간 수준 화면)": {"longest_gap_s": 6, "stop_button": False, "confirm_before_irreversible": False, "summary_lists_actions": True},
    "고친 화면": {"longest_gap_s": 3, "stop_button": True, "confirm_before_irreversible": True, "summary_lists_actions": True},
}


def main():
    ct = crosstab(HE_TASKS)
    print("[1] He 외 과제 구성(서술 옮김) 교차표 (위험, 계획이 맞음): " + ", ".join("%s/%s=%d" % (r, "맞음" if ok else "틀림", v) for (r, ok), v in ct.items()))
    print("  빈 칸 %d개, 균형 %s" % (len(empty_cells(HE_TASKS)), "예" if is_balanced(HE_TASKS) else "아니요"))
    bal = balanced_design(6)
    print("  칸당 6단계 설계: 합 %d단계, 빈 칸 %d개, 균형 %s" % (len(bal), len(empty_cells(bal)), "예" if is_balanced(bal) else "아니요"))
    print("[2] 가상 감독 기록(24단계)")
    for name, spec in LOG_SPEC.items():
        m = metrics(expand(spec))
        print("  %s: 오류 %d단계 중 잡음 %d(%.1f%%), 맞은 %d단계 중 괜히 멈춤 %d(%.1f%%), 판단이 맞은 비율 %.1f%%" % (
            name, m["wrong"], m["caught"], m["catch_rate"], m["right"], m["false_alarm"], m["false_alarm_rate"], m["decision_accuracy"]))
    m = metrics(expand(LOG_SPEC["화면 P"]), "낮음")
    h = metrics(expand(LOG_SPEC["화면 P"]), "높음")
    print("  화면 P를 위험별로 보면 잡은 비율 낮음 %.1f%%, 높음 %.1f%%" % (m["catch_rate"], h["catch_rate"]))
    print("[3] 화면 점검표(팀 한계 %d초, 가상)" % TEAM_GAP_LIMIT_S)
    for name, spec in SCREENS.items():
        res = checklist(spec)
        print("  %s: %d/%d 통과, 미통과 = %s" % (name, sum(res.values()), len(res), ", ".join(k for k, v in res.items() if not v) or "없음"))


if __name__ == "__main__":
    main()
