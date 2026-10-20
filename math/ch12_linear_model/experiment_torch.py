"""12장 · 선형 모델: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    x=torch.tensor(2.); y=torch.tensor(5.); w=torch.tensor(1.,requires_grad=True); pred=w*x; loss=(pred-y)**2; loss.backward(); new=w-.1*w.grad
    return {'prediction': float(pred.detach()), 'gradient': float(w.grad), 'new_weight': round(float(new.detach()),1), 'new_prediction': round(float((new*x).detach()),1)}

if __name__ == "__main__":
    print(compute())
