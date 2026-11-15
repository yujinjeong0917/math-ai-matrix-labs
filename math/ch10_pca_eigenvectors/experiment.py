"""10장 · PCA와 고유벡터: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: 점들이 가장 길게 퍼진 방향(첫째 주성분)을 특잇값 분해(SVD)로 찾는다. 점 (1,1), (2,2), (3,3)은 대각선 위에 있다.
입력 없음. 3 x 2 데이터 행렬이 고정돼 있다.
출력: {"principal_direction": [0.707, 0.707]}. 대각선 방향 1/sqrt(2) = 0.707이다. 장 본문 숫자와의 대응은 README로 확인 필요.
실행하면 결과를 results/ch10.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # 열마다 평균을 빼서 중심을 원점으로 옮긴 뒤 SVD. vt의 첫 행이 첫째 주성분 방향
    # 특이벡터는 부호가 임의(+-)라 절댓값을 취하고 소수 셋째 자리로 반올림해 비교 가능하게 만든다
    X=np.array([[1.,1.],[2.,2.],[3.,3.]])
    _,_,vt=np.linalg.svd(X-X.mean(0))
    return {'principal_direction': np.abs(np.round(vt[0],3)).tolist()}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch10.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
