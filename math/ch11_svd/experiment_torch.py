"""11장 · SVD: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: 대각 꼴 행렬의 특잇값(행렬이 방향마다 늘리는 크기)을 구한다.
출력: {"singular_values": [3.0, 2.0]}. 장 본문 숫자와의 대응은 README로 확인 필요.
"""
import torch

def compute():
    # svdvals는 특잇값만 계산한다(내림차순). 소수 여섯째 자리 반올림은 float32 잡음 제거용
    A=torch.tensor([[3.,0.,0.],[0.,2.,0.]])
    return {'singular_values': [round(float(v),6) for v in torch.linalg.svdvals(A)]}

if __name__ == "__main__":
    print(compute())
