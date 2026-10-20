"""9장 · 분산과 공분산: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    a=torch.tensor([0.,2.]); b=torch.tensor([1.,1.])
    return {'variance_a': float(torch.var(a,correction=0)), 'variance_b': float(torch.var(b,correction=0))}

if __name__ == "__main__":
    print(compute())
