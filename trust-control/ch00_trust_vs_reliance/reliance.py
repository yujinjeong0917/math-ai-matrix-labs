"""0장 실습: 믿음(태도)이 아니라 따름(행동)을 기록에서 세고, 의존율, 적정 신뢰, 과신, 과소신뢰를 계산한다.

모든 숫자는 가상의 예시이다(가상 쇼핑몰의 AI 답변 도우미, 답변 10건). 신뢰라는 태도는 기록에서 읽을 수 없고,
기록에 남는 것은 사용자가 답변을 따랐는지(행동)뿐이다. 표준 라이브러리만 쓴다.

정의(이 장이 쓰는 정의이다. 분모는 이 장이 정했다):
  의존율      = 따른 횟수 / 전체 횟수
  적정 신뢰   = (AI가 맞았을 때 따른 횟수 + AI가 틀렸을 때 따르지 않은 횟수) / 전체 횟수
  과신        = AI가 틀렸는데 따른 횟수 / AI가 틀린 횟수
  과소신뢰    = AI가 맞았는데 따르지 않은 횟수 / AI가 맞은 횟수
"""

from __future__ import annotations

# (AI 답변이 맞았나, 사용자가 따랐나). 가상 답변 10건: AI는 7건 맞고 3건 틀렸다.
AI_CORRECT = [True, True, True, True, True, True, True, False, False, False]

USERS = {
    # 맹목 추종: 10건 모두 따름
    "A": [True] * 10,
    # 가려서 따름: 맞은 7건 중 6건 따르고, 틀린 3건 중 1건 따름
    "B": [True, True, True, True, True, True, False, True, False, False],
    # 전부 거부: 하나도 따르지 않음
    "C": [False] * 10,
}


def metrics(ai_correct: list[bool], followed: list[bool]) -> dict:
    if len(ai_correct) != len(followed):
        raise ValueError("길이가 다르다")
    n = len(ai_correct)
    n_right = sum(ai_correct)
    n_wrong = n - n_right
    follow_right = sum(1 for c, f in zip(ai_correct, followed) if c and f)
    follow_wrong = sum(1 for c, f in zip(ai_correct, followed) if (not c) and f)
    skip_right = n_right - follow_right
    skip_wrong = n_wrong - follow_wrong
    return {
        "n": n,
        "reliance": (follow_right + follow_wrong) / n,
        "appropriate": (follow_right + skip_wrong) / n,
        "overtrust": follow_wrong / n_wrong if n_wrong else None,
        "undertrust": skip_right / n_right if n_right else None,
        "counts": {"맞았고 따름": follow_right, "맞았는데 안 따름": skip_right,
                   "틀렸는데 따름": follow_wrong, "틀렸고 안 따름": skip_wrong},
    }


def pct(x) -> str:
    return "없음" if x is None else f"{x * 100:.1f}%"


def main() -> None:
    print(f"[1] 가상 답변 {len(AI_CORRECT)}건 중 AI가 맞은 것 {sum(AI_CORRECT)}건 "
          f"(AI 정확도 {pct(sum(AI_CORRECT) / len(AI_CORRECT))})")
    for name, f in USERS.items():
        m = metrics(AI_CORRECT, f)
        print(f"[2] 사용자 {name}: 의존율 {pct(m['reliance'])}, 적정 신뢰 {pct(m['appropriate'])}, "
              f"과신 {pct(m['overtrust'])}, 과소신뢰 {pct(m['undertrust'])}, 칸 {m['counts']}")


if __name__ == "__main__":
    main()
