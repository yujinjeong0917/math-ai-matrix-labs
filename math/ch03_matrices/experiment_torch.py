"""3장 · 행렬: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: 행렬(행 = 글 하나, 열 = 특징)에 가중치 벡터를 곱해 행마다 점수를 한꺼번에 만든다.
출력: {"scores": [10, 11], "shape": [2, 2]}. 장 본문 숫자와의 대응은 README로 확인 필요.
"""
import torch

def compute():
    # numpy와 같은 연산(@)이다. 정수 텐서라 결과도 정수(int64)로 나와 numpy 결과와 정확히 같다
    X=torch.tensor([[2,1],[1,2]]); w=torch.tensor([3,4])
    return {'scores': (X@w).tolist(), 'shape': list(X.shape)}

if __name__ == "__main__":
    print(compute())
