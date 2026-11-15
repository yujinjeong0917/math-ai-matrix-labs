"""10장 · PCA와 고유벡터: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: 점들이 가장 길게 퍼진 방향(첫째 주성분)을 특잇값 분해(SVD)로 찾는다. 점 (1,1), (2,2), (3,3)은 대각선 위에 있다.
출력: {"principal_direction": [0.707, 0.707]}. 대각선 방향 1/sqrt(2) = 0.707이다. 장 본문 숫자와의 대응은 README로 확인 필요.
"""
import torch

def compute():
    # numpy와 같은 절차: 중심화(X - 열 평균) 후 svd. vh[0]이 첫째 주성분 방향. 부호는 임의라 절댓값을 쓴다
    X=torch.tensor([[1.,1.],[2.,2.],[3.,3.]]); _,_,vh=torch.linalg.svd(X-X.mean(0))
    return {'principal_direction': [round(abs(float(v)),3) for v in vh[0]]}

if __name__ == "__main__":
    print(compute())
