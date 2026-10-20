"""3장 · 행렬: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    X=torch.tensor([[2,1],[1,2]]); w=torch.tensor([3,4])
    return {'scores': (X@w).tolist(), 'shape': list(X.shape)}

if __name__ == "__main__":
    print(compute())
