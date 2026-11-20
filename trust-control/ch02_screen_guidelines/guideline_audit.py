"""2장 실습: 가상 답변 도우미 화면 3개를 지침 6개로 점검하고, 논문이 보고한 평정 개수의 합과 비율을 확인한다.

화면과 평정은 가상의 예시이다. 점검 결과는 화면에 그 요소가 있는지 없는지를 센 것이고,
그 요소가 사용자의 신뢰나 성과를 바꾸는지는 이 코드가 알 수 없다. 표준 라이브러리만 쓴다.

지침 번호와 뜻은 Amershi 외(2019)의 번호를 따르되, 문구는 이 장의 말로 옮겼다.
  1 무엇을 할 수 있는지 알린다   2 얼마나 잘하는지(얼마나 틀리는지) 알린다
  8 답변을 쉽게 닫을 수 있다     9 틀린 답을 쉽게 고칠 수 있다
 10 확신이 없으면 서비스 범위를 줄인다   11 왜 그렇게 답했는지 볼 수 있다
"""

from __future__ import annotations

GUIDELINES = {
    1: "무엇을 할 수 있는지 알린다",
    2: "얼마나 잘하는지 알린다",
    8: "답변을 쉽게 닫을 수 있다",
    9: "틀린 답을 쉽게 고칠 수 있다",
    10: "확신이 없으면 범위를 줄인다",
    11: "왜 그렇게 답했는지 볼 수 있다",
}

# True 적용, False 위반, None 해당 없음
SCREENS = {
    "화면 1 (답변만)": {1: False, 2: False, 8: True, 9: False, 10: False, 11: False},
    "화면 2 (답변 + 근거 문장)": {1: True, 2: False, 8: True, 9: False, 10: False, 11: True},
    "화면 3 (근거 + 정확도 안내 + 오류 신고 + 상담원 연결)": {1: True, 2: True, 8: True, 9: True, 10: True, 11: True},
}

# Amershi 외(2019)가 정리 후 보고한 평정 예시 개수(본문 서술)
PAPER_COUNTS = {"적용": 313, "위반": 277, "중립": 89, "해당 없음": 106}
PAPER_TOTAL = 785


def rate(features: dict) -> dict:
    out = {}
    for g in GUIDELINES:
        v = features.get(g)
        out[g] = "해당 없음" if v is None else ("적용" if v else "위반")
    return out


def tally(ratings: dict) -> dict:
    t = {"적용": 0, "위반": 0, "해당 없음": 0}
    for r in ratings.values():
        t[r] += 1
    return t


def paper_shares() -> dict:
    total = sum(PAPER_COUNTS.values())
    if total != PAPER_TOTAL:
        raise ValueError("논문 개수의 합이 785가 아니다")
    return {k: v / total for k, v in PAPER_COUNTS.items()}


def main() -> None:
    for name, feats in SCREENS.items():
        r = rate(feats)
        t = tally(r)
        print(f"[1] {name}: 적용 {t['적용']}, 위반 {t['위반']}, 해당 없음 {t['해당 없음']} "
              f"(위반: {[g for g, x in r.items() if x == '위반']})")
    print(f"[2] 논문 보고 개수 합 {' + '.join(str(v) for v in PAPER_COUNTS.values())} = {sum(PAPER_COUNTS.values())}")
    s = paper_shares()
    print("[3] 비율 " + ", ".join(f"{k} {v * 100:.1f}%" for k, v in s.items()))
    print("[4] 이 점검은 요소가 있는지만 센다. 신뢰나 성과가 달라지는지는 재지 않았다.")


if __name__ == "__main__":
    main()
