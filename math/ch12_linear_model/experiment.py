"""12장 · 선형 모델: 손계산과 같은 숫자를 만드는 결정적 실험."""
from pathlib import Path
import json
import numpy as np

def compute():
    x,y,w,lr=2.,5.,1.,.1; pred=w*x; grad=2*(pred-y)*x; new=w-lr*grad
    return {'prediction': pred, 'gradient': grad, 'new_weight': new, 'new_prediction': new*x}

if __name__ == "__main__":
    result = compute()
    target = Path(__file__).with_name("results") / "ch12.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
