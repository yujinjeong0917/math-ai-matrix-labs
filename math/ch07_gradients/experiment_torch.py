"""7장 · 그래디언트: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: 손실의 그래디언트(각 성분의 기울기 벡터)를 받아 경사하강 한 걸음 p - lr x g를 계산한다.
출력: {"gradient": [4.0, 6.0], "updated": [3.6, 3.4]}. 장 본문 숫자와의 대응은 README로 확인 필요.
"""
import torch

def compute():
    # 손실 = sum((p - [2, 1])^2)의 그래디언트는 2(p - [2, 1]) = [4, 6]이다(p = [4, 4]에서)
    # 한 걸음 계산 뒤 소수 첫째 자리로 반올림한다. torch는 기본 float32라 3.6, 3.4가 정확히 안 떨어지는 오차를 지우는 용도
    p=torch.tensor([4.,4.],requires_grad=True); loss=((p-torch.tensor([2.,1.]))**2).sum(); loss.backward(); q=p-.1*p.grad
    return {'gradient': p.grad.tolist(), 'updated': [round(v,1) for v in q.tolist()]}

if __name__ == "__main__":
    print(compute())
