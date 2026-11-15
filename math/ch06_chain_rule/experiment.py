"""6장 · 연쇄법칙: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: 합성함수 y = z^2, z = 3x의 x = 2에서의 미분을 연쇄법칙(dy/dx = dy/dz x dz/dx)으로 구한다.
입력 없음. x = 2가 고정돼 있다.
출력: {"x": 2.0, "z": 6.0, "dy_dx": 36.0}. 장 본문 숫자와의 대응은 README로 확인 필요.
실행하면 결과를 results/ch06.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # z = 3x = 6. dy/dz = 2z = 12, dz/dx = 3이므로 dy/dx = (2z) x 3 = 36
    x=2.; z=3*x
    return {'x': x, 'z': z, 'dy_dx': (2*z)*3}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch06.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
