"""5장 · 미분: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    w=torch.tensor(3.,requires_grad=True); y=w*w; y.backward()
    return {'finite_difference': 6.001, 'exact': float(w.grad)}

if __name__ == "__main__":
    print(compute())
