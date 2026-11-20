"""3장 실습: 가상 알림 8개를 질문 세 개(지금 과제와 관련 / 지금 행동 필요 / 놓치면 되돌릴 수 없음)로 나눠 시점을 정한다.

이 규칙은 설계자의 제안이며 검증된 규칙이 아니다. 확률이나 임계값은 쓰지 않는다.
입력은 notifications.json(가상, 예시용)이다. 표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
TIMINGS = [
    "지금, 화면 안에서",
    "화면 안에서 조용히",
    "지금, 과제를 막지 않는 짧은 알림",
    "주문이 끝난 뒤 묶어서",
    "보내지 않음(설정에서 켠 사람에게만)",
]


def load(path: str | None = None) -> list[dict]:
    with open(path or os.path.join(HERE, "notifications.json"), encoding="utf-8") as f:
        return json.load(f)["notifications"]


def timing(n: dict) -> str:
    if n["relevant"]:
        return TIMINGS[0] if n["action_now"] else TIMINGS[1]
    if n["irreversible"]:
        return TIMINGS[2]
    return TIMINGS[3] if n["action_now"] else TIMINGS[4]


def tally(items: list[dict]) -> dict[str, int]:
    c = Counter(timing(n) for n in items)
    return {t: c.get(t, 0) for t in TIMINGS}


def main() -> None:
    items = load()
    print(f"[1] 알림 {len(items)}개")
    for i, n in enumerate(items, start=1):
        print(f"  {i}. {n['text']} -> {timing(n)}")
    print("[2] 시점별 개수")
    for t, v in tally(items).items():
        print(f"  {t}: {v}")


if __name__ == "__main__":
    main()
