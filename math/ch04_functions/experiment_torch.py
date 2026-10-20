"""4장 · 함수: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    xs=torch.tensor([0,1,2])
    return {'inputs': xs.tolist(), 'outputs': (2*xs+1).tolist()}

if __name__ == "__main__":
    print(compute())
