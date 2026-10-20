"""7장 · 그래디언트: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    p=torch.tensor([4.,4.],requires_grad=True); loss=((p-torch.tensor([2.,1.]))**2).sum(); loss.backward(); q=p-.1*p.grad
    return {'gradient': p.grad.tolist(), 'updated': [round(v,1) for v in q.tolist()]}

if __name__ == "__main__":
    print(compute())
