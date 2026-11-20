"""5장 실습: 가상 쇼핑몰 답변 도우미의 두 화면(답변만 A, 답변과 근거 문장 B)을 비교하는 집계, 구간, 가이드 표 점검 (가상 데이터).

모든 숫자는 가상의 예시이다. 행동 기록은 손으로 정한 값이고 사람의 신뢰나 행동을 시뮬레이션한 것이 아니다.
참가자마다 AI 답변 16건을 본다(맞는 답변 8건, 틀린 답변 8건으로 설계). 기록은 맞는 답변 중 따른 건수(fr)와 틀린 답변 중 따른 건수(fw)이다.
  과신 = fw / 8,  과소신뢰 = (8 - fr) / 8,  적정 신뢰 = (fr + (8 - fw)) / 16
화면 비교는 참가자별 값의 화면 평균 차이이고, 95% 구간은 참가자를 다시 뽑는 부트스트랩(백분위 방식, 시드 고정)이다.
표준 라이브러리만 쓴다.
"""

from __future__ import annotations

import random
import statistics

N_RIGHT = 8
N_WRONG = 8

# (fr, fw), 화면별 참가자 12명(가상)
ARM_A = [(6, 4), (7, 5), (5, 3), (6, 4), (7, 5), (6, 4), (5, 3), (7, 5), (6, 4), (6, 3), (5, 4), (7, 5)]
ARM_B = [(8, 6), (7, 5), (8, 6), (7, 5), (8, 6), (7, 6), (8, 5), (7, 5), (8, 6), (7, 5), (8, 6), (8, 6)]


def person_measures(fr: int, fw: int) -> dict:
    if not (0 <= fr <= N_RIGHT and 0 <= fw <= N_WRONG):
        raise ValueError("건수가 범위를 벗어났다")
    return {
        "overtrust": fw / N_WRONG,
        "undertrust": (N_RIGHT - fr) / N_RIGHT,
        "appropriate": (fr + (N_WRONG - fw)) / (N_RIGHT + N_WRONG),
    }


def arm_values(arm: list, key: str) -> list:
    return [person_measures(fr, fw)[key] for fr, fw in arm]


def diff_of_means(a: list, b: list) -> float:
    return statistics.fmean(b) - statistics.fmean(a)


def bootstrap_ci(a: list, b: list, reps: int = 2000, seed: int = 20) -> tuple[float, float]:
    rng = random.Random(seed)
    diffs = []
    for _ in range(reps):
        ra = [a[rng.randrange(len(a))] for _ in a]
        rb = [b[rng.randrange(len(b))] for _ in b]
        diffs.append(diff_of_means(ra, rb))
    diffs.sort()
    return diffs[int(0.025 * reps)], diffs[int(0.975 * reps) - 1]


STRENGTHS = [
    "한 연구에서 측정",
    "한 연구에서 측정(원인 분리 안 됨)",
    "개념 제안, 효과 미측정",
    "읽은 논문이 측정하지 않음",
]
COLUMNS = ["조건", "대상", "이유", "근거", "강도"]

GUIDE_ROWS = [
    ["설명을 붙였더니 이해 문항 정답이 일부 설명에서 올랐다고 보고됨", "미국 거주자, 온라인 작업 플랫폼",
     "설명이 모델 행동을 보게 해서(저자 설명)", "Wang & Yin 실험 1", "한 연구에서 측정"],
    ["영역 지식이 적다고 본 과제에서 어떤 설명도 신뢰 보정과 불확실성 인지를 돕지 못함", "위와 같은 표본",
     "낯선 과제에서 설명을 읽는 부담이 커서(저자의 추측)", "Wang & Yin 실험 2", "한 연구에서 측정(원인 분리 안 됨)"],
    ["신경망 모델에서 LIME 기여도는 과소신뢰를 줄였지만 과신이 늘었다고 보고됨(회귀 계수)", "위와 같은 표본",
     "LIME이 모델 예측과 어긋나는 국소 근사라서(저자 설명)", "Wang & Yin 실험 3", "한 연구에서 측정"],
    ["화면에서 왜 그렇게 답했는지 볼 수 있게 하라는 지침(11번). 평정에서는 위반이 가장 많은 지침 중 하나였고 신뢰 보정 효과는 재지 않음", "미국 상위 앱 20개를 평가한 HCI 경험자 49명",
     "설명이 필요한데 찾지 못하는 경우가 많아서(저자 해석)", "Amershi 외", "개념 제안, 효과 미측정"],
    ["대화형 대규모 언어 모델 답변에서 설명이 신뢰 보정을 돕는다", "쇼핑몰 고객", "알 수 없음",
     "읽은 논문의 과제와 모델이 다르다", "읽은 논문이 측정하지 않음"],
    ["설명이 길수록(양이 많을수록) 신뢰 보정이 좋아진다", "쇼핑몰 고객", "알 수 없음",
     "읽은 실험은 설명의 종류를 비교했고 양을 바꾸지 않았다", "읽은 논문이 측정하지 않음"],
    ["정확도 안내 문구가 과신을 줄인다", "쇼핑몰 고객", "알 수 없음",
     "지침으로 제안됐을 뿐 효과를 잰 결과를 읽지 않았다", "읽은 논문이 측정하지 않음"],
]


def valid_guide_row(row: list) -> bool:
    return len(row) == len(COLUMNS) and all(isinstance(c, str) and c.strip() for c in row) and row[4] in STRENGTHS


def main() -> None:
    print(f"[1] 가상 참가자: 화면 A {len(ARM_A)}명, 화면 B {len(ARM_B)}명, 각자 AI 답변 {N_RIGHT + N_WRONG}건")
    for key, label in [("overtrust", "과신"), ("undertrust", "과소신뢰"), ("appropriate", "적정 신뢰")]:
        a, b = arm_values(ARM_A, key), arm_values(ARM_B, key)
        lo, hi = bootstrap_ci(a, b)
        print(f"[2] {label}: A 평균 {statistics.fmean(a) * 100:.1f}%, B 평균 {statistics.fmean(b) * 100:.1f}%, "
              f"차이(B-A) {diff_of_means(a, b) * 100:.1f}%p, 95% 구간 [{lo * 100:.1f}, {hi * 100:.1f}]%p")
    ok = sum(valid_guide_row(r) for r in GUIDE_ROWS)
    print(f"[3] 가이드 표 {len(GUIDE_ROWS)}줄 검사: 통과 {ok}줄")


if __name__ == "__main__":
    main()
