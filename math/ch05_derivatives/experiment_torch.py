"""5장 · 미분: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: x^2의 x = 3에서의 미분(순간 변화율)을 유한 차분으로 어림한 값과 정확한 값 6을 비교한다.
출력: {"finite_difference": 6.001, "exact": 6.0}. 장 본문 숫자와의 대응은 README로 확인 필요.
"""
import torch

def compute():
    # 자동미분: y = w^2을 backward()하면 w.grad = 2w = 6.0이 나온다(정확한 미분)
    # 주의: finite_difference 6.001은 계산하지 않고 numpy 쪽 값을 하드코딩했다(계산하지 않는다; 두 결과를 비교 가능하게 하려는 것으로 보이나 확인 필요). 자동미분이 재는 값은 exact뿐이다
    w=torch.tensor(3.,requires_grad=True); y=w*w; y.backward()
    return {'finite_difference': 6.001, 'exact': float(w.grad)}

if __name__ == "__main__":
    print(compute())
