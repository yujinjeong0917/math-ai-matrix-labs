"""4장 · 함수: 손계산과 같은 숫자를 만드는 결정적 실험."""
from pathlib import Path
import json
import numpy as np

def compute():
    xs=np.array([0,1,2])
    return {'inputs': xs.tolist(), 'outputs': (2*xs+1).tolist()}

if __name__ == "__main__":
    result = compute()
    target = Path(__file__).with_name("results") / "ch04.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
