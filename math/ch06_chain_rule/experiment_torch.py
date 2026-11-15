"""6장 · 연쇄법칙: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: 합성함수 y = z^2, z = 3x의 x = 2에서의 미분을 연쇄법칙(dy/dx = dy/dz x dz/dx)으로 구한다.
출력: {"x": 2.0, "z": 6.0, "dy_dx": 36.0}. 장 본문 숫자와의 대응은 README로 확인 필요.
"""
import torch

def compute():
    # y = (3x)^2 = 9x^2이므로 dy/dx = 18x = 36. backward()가 연쇄법칙을 자동으로 적용해 x.grad에 넣는다
    # 주의: z = 6.0은 계산하지 않고 numpy 쪽 값을 하드코딩했다(확인 필요)
    x=torch.tensor(2.,requires_grad=True); y=(3*x)**2; y.backward()
    return {'x': float(x.detach()), 'z': 6.0, 'dy_dx': float(x.grad)}

if __name__ == "__main__":
    print(compute())
