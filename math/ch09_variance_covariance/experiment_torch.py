"""9장 · 분산과 공분산: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: 평균에서 벗어난 정도를 재는 분산을 모분산(분모 n) 기준으로 계산한다. 값이 퍼져 있으면 크고 모두 같으면 0이다.
출력: {"variance_a": 1.0, "variance_b": 0.0}. 장 본문 숫자와의 대응은 README로 확인 필요.
"""
import torch

def compute():
    # 함정: torch.var의 기본은 표본분산(분모 n - 1)이다. correction=0으로 모분산에 맞춰야 numpy 값과 같다
    a=torch.tensor([0.,2.]); b=torch.tensor([1.,1.])
    return {'variance_a': float(torch.var(a,correction=0)), 'variance_b': float(torch.var(b,correction=0))}

if __name__ == "__main__":
    print(compute())
