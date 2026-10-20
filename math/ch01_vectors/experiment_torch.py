"""1장 · 벡터: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    a=torch.tensor([2,1,3]); b=torch.tensor([1,2,1])
    return {'sum': (a+b).tolist(), 'dimension': a.numel()}

if __name__ == "__main__":
    print(compute())
