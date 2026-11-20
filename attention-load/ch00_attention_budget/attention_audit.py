"""0장 실습: 같은 주문 입력 화면이 두 사용자에게 서로 다른 '지금 필요한 요소'로 나뉘는 것을 센다.

입력은 order_screen.json(가상, 예시용)이다. 분류는 설계자가 손으로 붙인 것이고 측정값이 아니다.
이 코드에는 '몇 개를 넘으면 많다' 같은 기준이 없다. 개수를 세기만 한다. 표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import json
import os
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ("결정", "확인", "무관")


def load(path: str | None = None) -> dict:
    with open(path or os.path.join(HERE, "order_screen.json"), encoding="utf-8") as f:
        return json.load(f)


def tally(data: dict, profile: str) -> dict[str, int]:
    """프로필 하나에서 결정/확인/무관 요소가 각각 몇 개인지 센다."""
    idx = data["profiles"].index(profile)
    c = Counter(e["need"][idx] for e in data["elements"])
    for k in c:
        if k not in CLASSES:
            raise ValueError(f"알 수 없는 분류: {k}")
    return {k: c.get(k, 0) for k in CLASSES}


def differing(data: dict) -> list[str]:
    """두 프로필에서 분류가 다른 요소 이름."""
    return [e["name"] for e in data["elements"] if len(set(e["need"])) > 1]


def both_irrelevant(data: dict) -> list[str]:
    return [e["name"] for e in data["elements"] if set(e["need"]) == {"무관"}]


def main() -> None:
    data = load()
    n = len(data["elements"])
    for i, p in enumerate(data["profiles"], start=1):
        t = tally(data, p)
        print(f"[{i}] {p}: 결정 {t['결정']}, 확인 {t['확인']}, 무관 {t['무관']} (합 {sum(t.values())})")
    diff = differing(data)
    print(f"[3] 두 사용자에서 분류가 다른 요소: {len(diff)}개 / {n}개 ({', '.join(diff)})")
    print(f"[4] 두 사용자 모두 '무관'인 요소: {', '.join(both_irrelevant(data))}")


if __name__ == "__main__":
    main()
