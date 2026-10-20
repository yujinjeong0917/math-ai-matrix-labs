"""8장 · 확률과 기댓값: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    v=torch.tensor([4.,0.]); p=torch.tensor([.25,.75])
    return {'probability_sum': float(p.sum()), 'expectation': float(v@p)}

if __name__ == "__main__":
    print(compute())
