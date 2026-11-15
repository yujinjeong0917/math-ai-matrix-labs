"""1장 · 벡터: 같은 입력을 PyTorch로 대조하는 실험.

무엇을 재는가: experiment.py(numpy)와 같은 계산을 PyTorch로 해서 결과가 같은지 대조한다. 이 파일은 torch가 필요하다.
대상: 벡터 두 개의 성분별 덧셈과 차원(성분 수)을 확인한다.
출력: {"sum": [3, 3, 4], "dimension": 3}. 장 본문 숫자와의 대응은 README로 확인 필요.
"""
import torch

def compute():
    # 텐서 합과 원소 수(numel)는 numpy의 a+b, a.size와 같은 값이다. tolist()로 파이썬 리스트로 바꿔 비교한다
    a=torch.tensor([2,1,3]); b=torch.tensor([1,2,1])
    return {'sum': (a+b).tolist(), 'dimension': a.numel()}

if __name__ == "__main__":
    print(compute())
