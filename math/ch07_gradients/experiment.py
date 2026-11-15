"""7장 · 그래디언트: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: 손실의 그래디언트(각 성분의 기울기 벡터)를 받아 경사하강 한 걸음 p - lr x g를 계산한다.
입력 없음. 현재 위치 p = [4, 4], 학습률 0.1이 고정돼 있다.
출력: {"gradient": [4.0, 6.0], "updated": [3.6, 3.4]}. 장 본문 숫자와의 대응은 README로 확인 필요.
실행하면 결과를 results/ch07.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # 그래디언트 g는 손으로 구한 값을 직접 넣었다([4, 6]). 같은 폴더 experiment_torch.py의 손실 sum((p - [2, 1])^2)에서 나오는 값이다
    # 한 걸음: q = p - 0.1 x g = [4 - 0.4, 4 - 0.6]. 그래디언트 반대 방향으로 이동한다
    p=np.array([4.,4.]); g=np.array([4.,6.]); q=p-.1*g
    return {'gradient': g.tolist(), 'updated': q.tolist()}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch07.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
