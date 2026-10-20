"""10장 · PCA와 고유벡터: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    X=torch.tensor([[1.,1.],[2.,2.],[3.,3.]]); _,_,vh=torch.linalg.svd(X-X.mean(0))
    return {'principal_direction': [round(abs(float(v)),3) for v in vh[0]]}

if __name__ == "__main__":
    print(compute())
