"""4장 · 함수: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: 입력 하나에 출력 하나를 내는 함수 f(x) = 2x + 1을 세 입력에서 계산한다.
입력 없음. xs = [0, 1, 2]가 코드에 고정돼 있다.
출력: {"inputs": [0, 1, 2], "outputs": [1, 3, 5]}. 장 본문 숫자와의 대응은 README로 확인 필요.
실행하면 결과를 results/ch04.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # f(x) = 2x + 1: 배열 전체에 한 번에 적용된다(원소별 연산)
    xs=np.array([0,1,2])
    return {'inputs': xs.tolist(), 'outputs': (2*xs+1).tolist()}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch04.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
