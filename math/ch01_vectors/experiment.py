"""1장 · 벡터: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: 벡터 두 개의 성분별 덧셈과 차원(성분 수)을 확인한다.
입력 없음. a = [2, 1, 3], b = [1, 2, 1]이 코드에 고정돼 있다.
출력: {"sum": [3, 3, 4], "dimension": 3}. 장 본문 숫자와의 대응은 README로 확인 필요.
실행하면 결과를 results/ch01.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # 같은 자리끼리 더한다: [2+1, 1+2, 3+1]. 두 벡터의 길이(차원)가 같아야 덧셈이 정의된다
    a=np.array([2,1,3]); b=np.array([1,2,1])
    return {'sum': (a+b).tolist(), 'dimension': int(a.size)}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch01.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
