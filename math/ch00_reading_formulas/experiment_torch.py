"""0장 · 식을 읽는 약속: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    result=(torch.tensor(2)*3)+(torch.tensor(1)*4)
    return {'result': int(result)}

if __name__ == "__main__":
    print(compute())
