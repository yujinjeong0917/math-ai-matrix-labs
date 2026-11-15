"""12장 · 선형 모델: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: 선형 모델 예측 pred = w x x의 제곱오차 손실을 경사하강 한 걸음 줄이고 예측이 목표에 가까워지는지 본다.
출력: {"prediction": 2.0, "gradient": -12.0, "new_weight": 2.2, "new_prediction": 4.4}. 장 본문 숫자와의 대응은 README로 확인 필요.
"""
import torch

def compute():
    # 손실 (pred - y)^2을 backward()하면 w.grad = 2(pred - y) x = -12
    # 한 걸음 뒤 값은 소수 첫째 자리로 반올림한다(float32 오차 제거). numpy와 같은 값이 나와야 한다
    x=torch.tensor(2.); y=torch.tensor(5.); w=torch.tensor(1.,requires_grad=True); pred=w*x; loss=(pred-y)**2; loss.backward(); new=w-.1*w.grad
    return {'prediction': float(pred.detach()), 'gradient': float(w.grad), 'new_weight': round(float(new.detach()),1), 'new_prediction': round(float((new*x).detach()),1)}

if __name__ == "__main__":
    print(compute())
