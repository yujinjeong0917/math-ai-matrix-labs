"""4장 · 함수: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: 입력 하나에 출력 하나를 내는 함수 f(x) = 2x + 1을 세 입력에서 계산한다.
출력: {"inputs": [0, 1, 2], "outputs": [1, 3, 5]}. 장 본문 숫자와의 대응은 README로 확인 필요.
"""
import torch

def compute():
    # 텐서에도 같은 식이 원소별로 적용된다
    xs=torch.tensor([0,1,2])
    return {'inputs': xs.tolist(), 'outputs': (2*xs+1).tolist()}

if __name__ == "__main__":
    print(compute())
