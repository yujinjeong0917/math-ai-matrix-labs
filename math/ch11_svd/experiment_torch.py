"""11장 · SVD: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    A=torch.tensor([[3.,0.,0.],[0.,2.,0.]])
    return {'singular_values': [round(float(v),6) for v in torch.linalg.svdvals(A)]}

if __name__ == "__main__":
    print(compute())
