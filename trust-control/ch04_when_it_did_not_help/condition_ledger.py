"""4장 실습: 실험들의 조건이 서로 어디가 다른지, 내 화면의 조건이 실험이 다룬 조건에 있는지 장부로 확인한다.

실험 조건은 Wang & Yin(2022) 본문 표에서 옮겼다(None은 읽은 범위에서 찾지 못한 칸). 가상 쇼핑몰 화면의 조건은 가상이다.
이 코드는 같은 값이 있는지만 센다. 비슷한지, 효과가 같을지는 판단하지 않는다. 표준 라이브러리만 쓴다.
"""

from __future__ import annotations

FIELDS = ["과제", "데이터", "특징 수", "모델", "모델 정확도(%)", "모델 AUC", "훈련 과제 수", "기본 보수(달러)"]

EXPERIMENTS = {
    1: dict(zip(FIELDS, ["재범 예측", "COMPAS", 7, "로지스틱 회귀", 69.1, 0.749, 0, 1.80])),
    2: dict(zip(FIELDS, ["산림 피복 예측", "UCI 산림 피복", 8, "로지스틱 회귀", 69.5, 0.686, 10, 2.00])),
    3: dict(zip(FIELDS, ["재범 예측", "COMPAS", 7, "신경망", 68.3, 0.761, None, None])),
}


def differing(a: int, b: int) -> tuple[list[str], list[str]]:
    """두 실험이 다른 칸 이름과, 한쪽이라도 값을 모르는 칸 이름을 돌려준다."""
    diff, unknown = [], []
    for f in FIELDS:
        x, y = EXPERIMENTS[a][f], EXPERIMENTS[b][f]
        if x is None or y is None:
            unknown.append(f)
        elif x != y:
            diff.append(f)
    return diff, unknown


# 실험들이 다룬 값의 모음(차원별)
COVERED = {
    "과제": {"재범 예측", "산림 피복 예측"},
    "모델": {"로지스틱 회귀", "신경망"},
    "출력 형식": {"두 범주 예측과 확률"},
    "설명 형식": {"특징 중요도", "특징 기여도", "최근접 이웃", "반사실 예시", "SHAP 기여도", "LIME 기여도"},
    "참가자": {"미국 거주자(온라인 작업 플랫폼)"},
    "과제 구성": {"같은 과제 32건을 연달아, 건별 정답 피드백 없음"},
    "재는 결과": {"객관적 이해", "주관적 이해", "확신별 의존", "적정 신뢰", "과신", "과소신뢰"},
}

SHOP_SCREEN = {  # 가상 쇼핑몰 답변 도우미 화면
    "과제": "쇼핑몰 문의 답변 확인",
    "모델": "대규모 언어 모델",
    "출력 형식": "자연어 답변",
    "설명 형식": "답변 근거 문장",
    "참가자": "쇼핑몰 고객",
    "과제 구성": "고객 한 명이 문의 한 건씩",
    "재는 결과": "적정 신뢰",
}


def coverage(deploy: dict, covered: dict = COVERED) -> tuple[list[str], list[str]]:
    """(실험이 다룬 차원, 다루지 않은 차원)."""
    hit, miss = [], []
    for dim, vals in covered.items():
        (hit if deploy.get(dim) in vals else miss).append(dim)
    return hit, miss


def main() -> None:
    d, u = differing(1, 2)
    print(f"[1] 실험 1과 2가 다른 칸 {len(d)}개: {d}")
    d, u = differing(1, 3)
    print(f"[2] 실험 1과 3이 다른 칸 {len(d)}개: {d} / 읽은 범위에서 값을 모르는 칸 {len(u)}개: {u}")
    hit, miss = coverage(SHOP_SCREEN)
    print(f"[3] 가상 쇼핑몰 화면: 차원 {len(COVERED)}개 중 실험이 다룬 것 {len(hit)}개 {hit}, 다루지 않은 것 {len(miss)}개 {miss}")


if __name__ == "__main__":
    main()
