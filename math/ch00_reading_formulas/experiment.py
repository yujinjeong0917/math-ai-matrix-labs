"""0장 · 식을 읽는 약속: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: 가중합 (2 x 3) + (1 x 4)를 파이썬 정수 연산으로 계산한다. 식을 읽는 약속(곱한 뒤 더한다)을 숫자로 확인하는 가장 작은 예다.
입력 없음(숫자는 코드 안에 고정).
출력: {"result": 10}. 장 본문의 손계산 2x3 + 1x4 = 10과 같다.
실행하면 결과를 results/ch00.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # 손계산 그대로: 2 x 3 = 6, 1 x 4 = 4, 합 10. numpy는 import하지만 이 장에서는 쓰지 않는다
    result = (2 * 3) + (1 * 4)
    return {'result': result}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch00.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
