"""1장 실습: 응답 지연 연구(Tan 외 2026)의 조건과 내 화면의 조건을 항목별로 대조한다.

숫자 중 논문 값은 논문 보고값이고, 가상이라고 적은 것은 이 장의 예시용이다.
이 코드는 "같은 조건인가"를 세는 점검표이지 효과를 예측하지 않는다. 표준 라이브러리만 쓴다.
"""
import math

# 논문 보고값(사려 깊음, 7점 척도): (평균, 표준편차)
THOUGHTFUL = {"2초": (5.76, 1.18), "9초": (6.09, 0.90), "20초": (6.11, 0.82)}
USEFUL_MEAN = {"2초": 6.19, "9초": 6.44}  # 20초 평균은 논문 본문에서 찾지 못함
CELL_N = {"창작": (41, 42, 41), "조언": (38, 38, 40)}  # 지연 2, 9, 20초 순서
LOG_N = 309
EXCLUDED_BEFORE = 6   # 동의 철회 3, 주의 검사 탈락 1, 속도 위반 2
EXCLUDED_BY_RULE = 63  # 관측된 첫 글자 지연이 배정값의 120%를 넘어 빠진 인원
SCALE_MAX = 7
# 본문에서 인용하는 그 밖의 논문 보고값(검산 대상이 아니라 기록용)
REPORTED = {
    "사려 깊음 p(2초-9초, 2초-20초)": (0.008, 0.040),
    "유용성 p(2초-9초)": 0.021,
    "편집한 사람 / 재생성한 사람(240명 중)": (13, 16),
    "편집, 재생성 비율(%)": (5.4, 6.7),
    "지연을 느꼈다고 답해 자유 응답을 분석한 인원": 140,
}

# 항목별 태그. 같은 태그면 같은 조건으로 센다.
STUDY = {
    "지연 종류": "첫 글자까지의 지연",
    "지연을 만든 방식": "서버가 출력을 붙잡음(인위적)",
    "대기 중 화면": "깜박이는 커서만",
    "과제": "창작과 조언(채팅 글쓰기, 추천)",
    "평가 방식": "과제 3개 뒤 품질 7문항 한 번",
    "참가자": "미국 온라인 패널, AI 비서 월 1회 이상 사용",
    "대기 길이(초)": (2, 20),
}

CASES = {
    "가상 환불 진행 화면": {
        "지연 종류": "여러 단계 전체 소요",
        "지연을 만든 방식": "실제 처리 시간",
        "대기 중 화면": "단계 목록과 진행 표시",
        "과제": "환불 처리(결과가 맞는지 확인 가능)",
        "평가 방식": "단계별 승인 기록",
        "참가자": "환불을 맡기는 상담 팀원",
        "대기 길이(초)": 20,
    },
    "가상 채팅 답변 도우미": {
        "지연 종류": "첫 글자까지의 지연",
        "지연을 만든 방식": "실제 처리 시간",
        "대기 중 화면": "깜박이는 커서만",
        "과제": "창작과 조언(채팅 글쓰기, 추천)",
        "평가 방식": "과제 3개 뒤 품질 7문항 한 번",
        "참가자": "일반 소비자",
        "대기 길이(초)": 9,
    },
}


def compare(case):
    """항목별로 (항목, 같은가) 목록을 돌려준다."""
    out = []
    for key, val in STUDY.items():
        if key == "대기 길이(초)":
            lo, hi = val
            out.append((key, lo <= case[key] <= hi))
        else:
            out.append((key, case[key] == val))
    return out


def n_same(case):
    return sum(1 for _, s in compare(case) if s)


def mean_diff(a, b):
    return round(a - b, 2)


def d_approx(g1, g2):
    """두 집단 크기가 같다고 가정한 근사 d(표준편차의 제곱 평균의 제곱근으로 나눔)."""
    (m1, s1), (m2, s2) = g1, g2
    return round(abs(m1 - m2) / math.sqrt((s1 ** 2 + s2 ** 2) / 2), 2)


def headroom(mean):
    return round(SCALE_MAX - mean, 2)


def total_n():
    return sum(sum(v) for v in CELL_N.values())


def excluded_share():
    return round(100 * EXCLUDED_BY_RULE / (LOG_N - EXCLUDED_BEFORE), 1)


def main():
    print("[1] 조건 대조(%d항목, 같으면 O)" % len(STUDY))
    for name, case in CASES.items():
        marks = " ".join("%s=%s" % (k, "O" if s else "X") for k, s in compare(case))
        print("  %s: 같은 항목 %d/%d" % (name, n_same(case), len(STUDY)))
        print("    " + marks)
    print("[2] 사려 깊음 평균 차이: 2초-9초 %.2f점, 2초-20초 %.2f점" % (
        mean_diff(THOUGHTFUL["9초"][0], THOUGHTFUL["2초"][0]), mean_diff(THOUGHTFUL["20초"][0], THOUGHTFUL["2초"][0])))
    print("[3] 7점 척도에서 위로 남은 칸: 2초 %.2f, 9초 %.2f" % (
        headroom(THOUGHTFUL["2초"][0]), headroom(THOUGHTFUL["9초"][0])))
    print("[4] 유용성 평균 차이: 2초-9초 %.2f점(20초 평균은 값 없음)" % mean_diff(USEFUL_MEAN["9초"], USEFUL_MEAN["2초"]))
    print("[5] 분석 인원 %d명(칸 6개 합)" % total_n())


if __name__ == "__main__":
    main()
