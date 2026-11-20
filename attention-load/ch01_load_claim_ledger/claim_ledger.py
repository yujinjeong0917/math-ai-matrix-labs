"""1장 실습: 화면 설계 주장을 '숫자 규칙', '절대 표현', 출처 표시로 나눠 센다.

입력은 claims.json(가상, 예시용)이다. 이 코드는 말투와 출처 표시만 본다. 주장이 맞는지는 판정하지 않는다.
숫자 규칙이나 절대 표현이 보이면 출처 표시와 상관없이 '읽은 근거 없음'으로 따로 센다.
표준 라이브러리만 쓴다.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
NUMERIC = re.compile(r"\d+\s*(개|%|초|자|줄|번|px)")
ABSOLUTE = re.compile(r"항상|반드시|절대|모든 사용자|누구에게나")
SOURCE_LABEL = {
    "리뷰가 소개한 틀": "이론 틀 소개 수준",
    "설계자 추측": "설계자 추측",
    "없음": "출처 없음",
}
ORDER = ["숫자 규칙: 읽은 근거 없음", "절대 표현: 읽은 근거 없음", "이론 틀 소개 수준", "설계자 추측", "출처 없음"]


def load(path: str | None = None) -> list[dict]:
    with open(path or os.path.join(HERE, "claims.json"), encoding="utf-8") as f:
        return json.load(f)["claims"]


def classify(text: str, source: str) -> str:
    if NUMERIC.search(text):
        return "숫자 규칙: 읽은 근거 없음"
    if ABSOLUTE.search(text):
        return "절대 표현: 읽은 근거 없음"
    if source not in SOURCE_LABEL:
        raise ValueError(f"알 수 없는 출처 표시: {source}")
    return SOURCE_LABEL[source]


def tally(claims: list[dict]) -> dict[str, int]:
    c = Counter(classify(x["text"], x["source"]) for x in claims)
    return {k: c.get(k, 0) for k in ORDER}


def main() -> None:
    claims = load()
    print(f"[1] 주장 {len(claims)}개")
    for i, x in enumerate(claims, start=1):
        print(f"  {i}. {x['text']} -> {classify(x['text'], x['source'])}")
    t = tally(claims)
    print("[2] 분류별 개수: " + ", ".join(f"{k} {v}" for k, v in t.items()))


if __name__ == "__main__":
    main()
