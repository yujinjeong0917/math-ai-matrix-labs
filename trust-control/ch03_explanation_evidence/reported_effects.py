"""3장 실습: Wang & Yin(2022)이 보고한 효과 크기와 95% 구간 17줄을 옮겨, 구간이 0을 포함하는지 센다.

값은 논문 본문에 적힌 것을 옮긴 것이다(그림의 막대는 읽지 못했다). 17줄은 이 장이 고른 것이지 논문의 전체 결과가 아니다.
d는 처치 집단과 통제 집단의 차이를 합동 표준편차로 나눈 효과 크기이고, 구간은 논문의 부트스트랩 95% 신뢰구간이다.
이 코드는 구간이 0을 포함하는지만 센다. 효과가 있는지 없는지를 판정하지 않는다. 표준 라이브러리만 쓴다.
"""

from __future__ import annotations

# (실험, 성질, 지표와 설명, d, 하한, 상한, 논문 표의 체크 표시 여부 또는 None)
ROWS = [
    (1, "이해", "객관적 이해, 특징 중요도", 0.26, 0.03, 0.48, None),
    (1, "이해", "객관적 이해, 반사실 예시", 0.27, 0.04, 0.48, None),
    (1, "이해", "주관적 이해, 네 설명 합친 값", 0.28, 0.05, 0.49, None),
    (1, "불확실성 인지", "확신에 따른 의존 차이, 네 설명 합친 값", 0.20, -0.02, 0.42, None),
    (1, "신뢰 보정", "적정 신뢰, 특징 중요도", 0.19, -0.05, 0.41, None),
    (1, "신뢰 보정", "적정 신뢰, 특징 기여도", 0.19, -0.03, 0.40, True),
    (1, "신뢰 보정", "적정 신뢰(확신 높은 예측만), 특징 중요도", 0.30, 0.07, 0.53, None),
    (1, "신뢰 보정", "적정 신뢰(확신 높은 예측만), 특징 기여도", 0.23, 0.02, 0.45, None),
    (1, "신뢰 보정", "과신(확신 낮은 예측만), 특징 중요도", 0.19, -0.03, 0.42, None),
    (2, "이해", "객관적 이해, 특징 중요도", 0.33, 0.06, 0.59, None),
    (2, "이해", "주관적 이해, 특징 기여도", 0.28, 0.01, 0.55, None),
    (3, "이해", "객관적 이해, SHAP 기여도", 0.30, 0.02, 0.57, None),
    (3, "이해", "주관적 이해, LIME 기여도", 0.22, -0.05, 0.50, None),
    (3, "이해", "주관적 이해, SHAP 기여도", 0.22, -0.06, 0.50, None),
    (3, "불확실성 인지", "확신에 따른 의존 차이, SHAP 기여도", 0.37, 0.08, 0.64, None),
    (3, "신뢰 보정", "적정 신뢰, SHAP 기여도", 0.38, 0.07, 0.65, None),
    (3, "신뢰 보정", "과소신뢰, SHAP 기여도", -0.33, -0.61, -0.01, None),
]

# 실험 2에서 본문이 값 없이 결론만 적은 것(이 코드는 숫자로 옮기지 못했다)
TEXT_ONLY = [
    (2, "불확실성 인지", "네 설명 모두 확신에 따른 의존 변화가 없었다고 적음"),
    (2, "신뢰 보정", "어떤 설명도 신뢰 보정을 돕지 못했다고 적음"),
]


def includes_zero(lo: float, hi: float) -> bool:
    return lo <= 0.0 <= hi


def midpoint(lo: float, hi: float) -> float:
    return (lo + hi) / 2


def tally(rows=ROWS) -> dict:
    out: dict = {}
    for exp, prop, _, _, lo, hi, _ in rows:
        key = (exp, prop)
        inc = includes_zero(lo, hi)
        a = out.setdefault(key, [0, 0])  # [구간이 0을 안 포함, 0을 포함]
        a[1 if inc else 0] += 1
    return out


def mismatches(rows=ROWS) -> list:
    """논문 표는 체크인데 d의 구간이 0을 포함하는 줄."""
    return [r for r in rows if r[6] is True and includes_zero(r[4], r[5])]


def widths(prop: str, rows=ROWS) -> tuple[float, float]:
    """한 성질의 줄들에서 구간 폭(상한-하한)의 최솟값과 최댓값."""
    w = [r[5] - r[4] for r in rows if r[1] == prop]
    return min(w), max(w)


def main() -> None:
    n_ex = sum(1 for r in ROWS if not includes_zero(r[4], r[5]))
    print(f"[1] 옮긴 줄 {len(ROWS)}개 중 구간이 0을 포함하지 않는 줄 {n_ex}개, 포함하는 줄 {len(ROWS) - n_ex}개")
    for (exp, prop), (ex, inc) in sorted(tally().items()):
        print(f"[2] 실험 {exp} {prop}: 0을 안 포함 {ex}줄, 포함 {inc}줄")
    for exp, prop, note in TEXT_ONLY:
        print(f"[3] 실험 {exp} {prop}: 값 없이 서술만 옮김({note})")
    for r in mismatches():
        print(f"[4] 표는 체크인데 d={r[3]:.2f}의 구간 [{r[4]:.2f}, {r[5]:.2f}]이 0을 포함: 실험 {r[0]} {r[2]}")
    worst = max(abs(midpoint(r[4], r[5]) - r[3]) for r in ROWS)
    print(f"[5] 구간 가운데와 d의 최대 차이 {worst:.3f}")
    lo, hi = widths("이해")
    print(f"[6] 이해 줄의 구간 폭: 최소 {lo:.2f}, 최대 {hi:.2f}")


if __name__ == "__main__":
    main()
