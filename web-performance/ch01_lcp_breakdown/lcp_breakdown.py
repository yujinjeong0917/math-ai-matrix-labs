"""1장 실습: LCP 한 값을 네 구간으로 나누고, 어느 요소가 LCP로 보고되는지 따라간다.

모든 시간은 가상, 예시용이다. 사람이 느낀 시간은 계산하지 않는다.
네 구간과 권장 비율(약 40%, 10% 미만, 약 40%, 10% 미만)은 web.dev 문서(optimize-lcp, 2026-11-20 확인)가
"지침이지 엄격한 규칙이 아니다"라고 적은 값이다. 목표값으로 쓰지 않는다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

PARTS = ("TTFB", "자원 로드 지연", "자원 로드 시간", "요소 렌더 지연")
GUIDE_SHARE = (40.0, 10.0, 40.0, 10.0)  # 퍼센트. 문서의 상대 지침
LCP_GOOD_S = 2.5

# 가상: 상품 목록 첫 화면의 대표 상품 이미지가 LCP인 방문 하나(초)
BEFORE = (0.9, 0.5, 0.9, 0.5)
# 가상: 이미지 파일을 줄였더니 로드 시간은 줄었지만 렌더 지연이 늘어난 경우
SHRUNK_IMAGE = (0.9, 0.5, 0.5, 0.9)
# 가상: 이미지를 첫 HTML에서 일찍 발견하게 하고 렌더 차단을 줄인 경우
EARLY_DISCOVERY = (0.9, 0.1, 0.5, 0.5)

# 가상: 같은 방문에서 "가장 큰 요소"가 바뀐 기록(초, 요소 이름)
CANDIDATES = [(1.2, "상품명 텍스트 블록"), (2.1, "배너 이미지"), (2.8, "대표 상품 이미지")]


def total(parts: tuple[float, ...]) -> float:
    return round(sum(parts), 3)


def shares(parts: tuple[float, ...]) -> list[float]:
    t = sum(parts)
    return [round(p / t * 100, 1) for p in parts]


def gap_from_guide(parts: tuple[float, ...]) -> list[float]:
    """실제 비율에서 지침 비율을 뺀 퍼센트포인트. 지침은 목표가 아니라 비교용이다."""
    return [round(s - g, 1) for s, g in zip(shares(parts), GUIDE_SHARE)]


def biggest_part(parts: tuple[float, ...]) -> str:
    top = max(parts)
    return " · ".join(PARTS[i] for i, p in enumerate(parts) if p == top)


def reported_lcp(candidates: list[tuple[float, str]], first_input_s: float | None = None) -> tuple[float, str]:
    """가장 최근의 후보를 보고한다. 사용자가 입력(탭, 스크롤, 키)하면 그 뒤 후보는 보고하지 않는다.
    입력 시각과 후보 시각이 같을 때의 처리는 문서 요약만으로 알 수 없어서 이 실습의 데이터는 같게 두지 않는다."""
    ok = [c for c in candidates if first_input_s is None or c[0] < first_input_s]
    if not ok:
        raise ValueError("입력 전에 그려진 후보가 없다")
    return ok[-1]


# 스스로 점검용 가상 값
QUIZ_PARTS = (0.8, 0.2, 1.0, 0.4)


def main() -> None:
    for name, parts in (("처음", BEFORE), ("이미지 줄임", SHRUNK_IMAGE), ("일찍 발견", EARLY_DISCOVERY)):
        t = total(parts)
        sh = shares(parts)
        print(f"[{name}] 비율 원값 {[round(p / sum(parts), 4) for p in parts]}")
        print(f"[{name}] 구간 {list(parts)} 합 {t}초 -> {'2.5초 이하' if t <= LCP_GOOD_S else '2.5초 초과'}")
        print(f"[{name}] 비율 {sh}%, 지침과의 차이 {gap_from_guide(parts)}%p, 가장 긴 구간 {biggest_part(parts)}")
    for label, stop in (("입력 없음", None), ("2.0초에 스크롤", 2.0), ("2.5초에 스크롤", 2.5)):
        t, who = reported_lcp(CANDIDATES, stop)
        print(f"[보고되는 LCP] {label}: {t}초 {who}")
    print(f"[점검 1] 구간 {list(QUIZ_PARTS)} 합 {total(QUIZ_PARTS)}초, TTFB 비율 {shares(QUIZ_PARTS)[0]}%")
    print(f"[점검 3] 2.3초에 탭하면 보고되는 LCP {reported_lcp(CANDIDATES, 2.3)[0]}초 {reported_lcp(CANDIDATES, 2.3)[1]}")


if __name__ == "__main__":
    main()
