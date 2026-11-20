"""0장 실습: 가상 환불 에이전트의 진행 화면에서 정보 수준별로 화면이 바뀌는 시각을 센다.

모든 숫자는 가상, 예시용이다. 여기서 센 것은 "화면이 바뀌는 시각"이고, 사람이 그 시간을
어떻게 느끼는지는 재지 않는다. 표준 라이브러리만 쓴다.
"""

TOTAL = 20  # 가상 환불 한 건의 처리 시간(초)

# (단계 이름, 시작, 끝) 초 단위, 가상
STEPS = [
    ("접수 해석", 0, 2),
    ("주문 조회", 2, 5),
    ("규정 확인", 5, 9),
    ("환불 지급", 9, 15),
    ("고객 안내", 15, 17),
    ("기록", 17, 20),
]

# Yu 외(2025)가 형성 연구에서 정한 정보 구성요소 10개를 환불 화면에 옮긴 이 장의 대응.
# 값은 (단계, 그 항목이 화면에서 바뀌는 시각들). 시각은 가상이다.
COMPONENTS = {
    "입력 해석": ("계획", [1]),
    "실행 계획": ("계획", [2]),
    "진행 중인 과제": ("실행", [0, 2, 5, 9, 15, 17]),
    "추론 과정": ("실행", [4, 8]),
    "실행 단계": ("실행", [5, 9, 15]),
    "중간 결과": ("실행", [5, 9, 15]),
    "과제 요약": ("요약", [20]),
    "결정 이유": ("요약", [20]),
    "점검 메모": ("요약", [20]),
    "다음 제안": ("요약", [20]),
}

# 논문의 세 수준: 높음은 10개 모두, 중간은 실행 계획, 진행 중인 과제, 과제 요약, 낮음은 과제 요약만.
LEVELS = {
    "높음": list(COMPONENTS),
    "중간": ["실행 계획", "진행 중인 과제", "과제 요약"],
    "낮음": ["과제 요약"],
}

# Yu 외(2025) 표 1의 평균(7점 척도, N=40). 순서는 높음, 중간, 낮음.
YU_MEANS = {
    "신뢰": (5.14, 4.88, 4.14),
    "만족": (5.33, 4.95, 3.98),
    "사용 의향": (5.23, 4.83, 4.03),
}
YU_N = 40
# 본문에서 인용하는 그 밖의 논문 보고값(검산 대상이 아니라 기록용)
YU_REPORTED = {
    "낮음 조건 표준편차(신뢰, 만족, 사용 의향)": (1.65, 1.95, 1.80),
    "높음 대 중간 쌍비교 p(신뢰, 만족, 사용 의향)": (0.026, 0.045, 0.016),
    "참가자 연령 19~35세, 여성 비율(%)": 77.5,
    "시연 영상 길이(분)": 2,
    "포스터 쪽수 379~384": 6,
}
YU_STUDENTS = 33
YU_PREFER_HIGH = 26
YU_PREFER_MID_STUDENTS = 12


def update_times(level):
    times = set()
    for name in LEVELS[level]:
        times.update(COMPONENTS[name][1])
    return sorted(times)


def gaps(times, start=0):
    """요청을 보낸 시각(start)부터 바뀌는 시각들 사이의 간격 목록."""
    out = []
    prev = start
    for t in times:
        if t > prev:
            out.append((t - prev, prev, t))
        prev = max(prev, t)
    return out


def longest_gap(times):
    return max(gaps(times))  # (길이, 시작, 끝)


def step_at(start, end):
    names = [n for n, s, e in STEPS if s < end and e > start]
    return names[0] if len(names) == 1 else "여러 단계"


def level_summary(level):
    times = update_times(level)
    length, a, b = longest_gap(times)
    return {
        "level": level,
        "items": len(LEVELS[level]),
        "moments": len(times),
        "times": times,
        "longest": length,
        "from": a,
        "to": b,
        "step": step_at(a, b),
    }


def with_extra(times, t):
    return sorted(set(times) | {t})


def diff(means, i, j):
    return round(means[i] - means[j], 2)


def share(x, n):
    return round(100 * x / n, 1)


def main():
    print("[1] 가상 환불 한 건 %d초, 단계 %d개" % (TOTAL, len(STEPS)))
    for n, s, e in STEPS:
        print("  %s %d~%d초" % (n, s, e))
    print("[2] 정보 수준별 화면")
    for lv in LEVELS:
        r = level_summary(lv)
        print("  %s: 항목 %d개, 바뀌는 시각 %d곳(%s초), 가장 긴 조용한 구간 %d초(%d초~%d초, %s)" % (
            lv, r["items"], r["moments"], ",".join(str(t) for t in r["times"]), r["longest"], r["from"], r["to"], r["step"]))
    print("[3] 환불 지급 중 12초에 문구를 하나 더하면 가장 긴 조용한 구간")
    for lv in ("높음", "중간"):
        base = update_times(lv)
        print("  %s: %d초 -> %d초" % (lv, longest_gap(base)[0], longest_gap(with_extra(base, 12))[0]))
    print("[4] Yu 외 표 1 평균 차이(높음-낮음 / 높음-중간)")
    for k, m in YU_MEANS.items():
        print("  %s: %.2f / %.2f" % (k, diff(m, 0, 2), diff(m, 0, 1)))
    print("[5] 가장 만족스러운 조건: 높음 %d명, 중간 학생 %d명, 설명 없는 인원 %d명" % (
        YU_PREFER_HIGH, YU_PREFER_MID_STUDENTS, YU_N - YU_PREFER_HIGH - YU_PREFER_MID_STUDENTS))
    print("[6] 중간을 고른 학생 12명은 40명의 %.1f%%, 학생 33명의 %.1f%%" % (
        share(YU_PREFER_MID_STUDENTS, YU_N), share(YU_PREFER_MID_STUDENTS, YU_STUDENTS)))


if __name__ == "__main__":
    main()
