"""11장 · SVD: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: 대각 꼴 행렬의 특잇값(행렬이 방향마다 늘리는 크기)을 구한다.
입력 없음. A = [[3, 0, 0], [0, 2, 0]]이 고정돼 있다.
출력: {"singular_values": [3.0, 2.0]}. 장 본문 숫자와의 대응은 README로 확인 필요.
실행하면 결과를 results/ch11.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # 대각 꼴이라 특잇값은 대각 성분 3, 2의 절댓값이고 큰 것부터 내림차순으로 나온다. 소수 여섯째 자리 반올림은 float 잡음 제거용
    A=np.array([[3.,0.,0.],[0.,2.,0.]])
    return {'singular_values': np.round(np.linalg.svd(A,compute_uv=False), 6).tolist()}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch11.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
