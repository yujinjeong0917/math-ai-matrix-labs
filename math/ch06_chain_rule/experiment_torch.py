"""6장 · 연쇄법칙: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    x=torch.tensor(2.,requires_grad=True); y=(3*x)**2; y.backward()
    return {'x': float(x.detach()), 'z': 6.0, 'dy_dx': float(x.grad)}

if __name__ == "__main__":
    print(compute())
