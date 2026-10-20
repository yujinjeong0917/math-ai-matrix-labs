"""7장 · 그래디언트: 손계산과 같은 숫자를 만드는 결정적 실험."""
from pathlib import Path
import json
import numpy as np

def compute():
    p=np.array([4.,4.]); g=np.array([4.,6.]); q=p-.1*g
    return {'gradient': g.tolist(), 'updated': q.tolist()}

if __name__ == "__main__":
    result = compute()
    target = Path(__file__).with_name("results") / "ch07.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
