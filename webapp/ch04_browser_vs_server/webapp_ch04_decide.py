"""기능마다 어디서 돌릴지 정하는 판단표.

규칙은 위에서부터 차례로 보고, 처음 걸리는 규칙에서 멈춘다.
1. 보이면 안 되는 것(비밀 키, 지시문, 원본 템플릿, 워터마크 판단)을 건드리면 서버.
   브라우저에서 돌리는 코드와 재료는 압축·난독화해도 사용자 컴퓨터에 와 있으니(1장) 지킬 수 없다.
2. 기기를 바꾸거나 브라우저 데이터가 지워져도 남아야 하면 서버(클라우드 DB, 5장).
3. 그 밖에는 브라우저. 사용자가 기다릴 수 있는 시간이 짧거나(0.1초 이하) 자주 쓰는 기능일수록
   서버로 보내면 요청 수와 대기 시간이 함께 늘어난다.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
FEATURES = HERE / "data" / "features.json"
INSTANT_MS = 100  # Nielsen: 0.1 second is about the limit for ... reacting instantaneously


def load_features():
    return json.loads(FEATURES.read_text(encoding="utf-8"))["features"]


def decide(feature):
    """한 기능의 (어디서, 걸린 규칙 번호, 이유)를 돌려준다."""
    if feature["touches"]:
        return "서버", 1, "보이면 안 되는 것을 씀: " + ", ".join(feature["touches"])
    if feature["must_survive_device_change"]:
        return "서버", 2, "기기를 바꾸거나 브라우저 데이터가 지워져도 남아야 함"
    if feature["budget_ms"] <= INSTANT_MS:
        return "브라우저", 3, f"{feature['budget_ms']}ms 안에 반응해야 하고 카드 한 장에 {feature['uses_per_card']}번 씀"
    return "브라우저", 3, "지킬 것이 없고, 서버로 보내면 요청과 비용만 늘어남"


def table(features=None):
    rows = []
    for f in features or load_features():
        where, rule, why = decide(f)
        rows.append({"id": f["id"], "name": f["name"], "where": where, "rule": rule, "why": why,
                     "uses_per_card": f["uses_per_card"], "budget_ms": f["budget_ms"]})
    return rows


if __name__ == "__main__":
    for r in table():
        print(f"{r['name']:<22} {r['where']:<5} 규칙{r['rule']}  {r['why']}")
