"""9장 · 분산과 공분산: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: 평균에서 벗어난 정도를 재는 분산을 모분산(분모 n) 기준으로 계산한다. 값이 퍼져 있으면 크고 모두 같으면 0이다.
입력 없음. a = [0, 2], b = [1, 1]이 고정돼 있다.
출력: {"variance_a": 1.0, "variance_b": 0.0}. 장 본문 숫자와의 대응은 README로 확인 필요.
실행하면 결과를 results/ch09.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # np.var는 기본이 모분산(분모 n, ddof=0). 평균 1에서 각각 1만큼 벗어나므로 (1 + 1) / 2 = 1.0
    a=np.array([0.,2.]); b=np.array([1.,1.])
    return {'variance_a': float(np.var(a)), 'variance_b': float(np.var(b))}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch09.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
