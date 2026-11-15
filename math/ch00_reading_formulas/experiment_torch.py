"""0장 · 식을 읽는 약속: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: 가중합 (2 x 3) + (1 x 4)를 파이썬 정수 연산으로 계산한다. 식을 읽는 약속(곱한 뒤 더한다)을 숫자로 확인하는 가장 작은 예다.
출력: {"result": 10}. 장 본문의 손계산 2x3 + 1x4 = 10과 같다.
"""
import torch

def compute():
    # 0차원 텐서끼리 곱하고 더한다. int()로 파이썬 정수로 바꿔야 numpy 쪽 결과(정수 10)와 == 비교가 맞는다
    result=(torch.tensor(2)*3)+(torch.tensor(1)*4)
    return {'result': int(result)}

if __name__ == "__main__":
    print(compute())
