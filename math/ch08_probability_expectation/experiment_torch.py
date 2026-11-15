"""8장 · 확률과 기댓값: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: 확률분포의 합이 1임을 확인하고 기댓값 E[X] = 값 x 확률의 합을 구한다.
출력: {"probability_sum": 1.0, "expectation": 1.0}. 장 본문 숫자와의 대응은 README로 확인 필요.
"""
import torch

def compute():
    # numpy와 같은 계산이다(float32). 0.25, 0.75는 이진수로 정확히 표현돼 합과 내적이 1.0과 정확히 같다
    v=torch.tensor([4.,0.]); p=torch.tensor([.25,.75])
    return {'probability_sum': float(p.sum()), 'expectation': float(v@p)}

if __name__ == "__main__":
    print(compute())
