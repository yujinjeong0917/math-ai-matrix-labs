"""12장 · 선형 모델: 손계산과 같은 숫자를 만드는 결정적 실험.

무엇을 재는가: 선형 모델 예측 pred = w x x의 제곱오차 손실을 경사하강 한 걸음 줄이고 예측이 목표에 가까워지는지 본다.
입력 없음. x = 2, 목표 y = 5, 가중치 w = 1, 학습률 0.1이 고정돼 있다.
출력: {"prediction": 2.0, "gradient": -12.0, "new_weight": 2.2, "new_prediction": 4.4}. 장 본문 숫자와의 대응은 README로 확인 필요.
실행하면 결과를 results/ch12.json에 저장한다. 이 파일은 numpy가 필요하다.
"""
from pathlib import Path
import json
import numpy as np

def compute():
    # 손실 L = (pred - y)^2의 w에 대한 기울기: dL/dw = 2(pred - y) x = 2(2 - 5) x 2 = -12
    # 기울기가 음수라 w를 키운다: w <- 1 - 0.1 x (-12) = 2.2. 새 예측 2.2 x 2 = 4.4로 목표 5에 가까워진다
    x,y,w,lr=2.,5.,1.,.1; pred=w*x; grad=2*(pred-y)*x; new=w-lr*grad
    return {'prediction': pred, 'gradient': grad, 'new_weight': new, 'new_prediction': new*x}

if __name__ == "__main__":
    result = compute()
    # 같은 폴더의 results/에 JSON으로 저장한다(폴더가 없으면 만든다). 사이트 본문 숫자와 대조하는 재현 기록
    target = Path(__file__).with_name("results") / "ch12.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
