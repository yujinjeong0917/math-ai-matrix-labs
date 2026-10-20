"""5장 · 미분: 손계산과 같은 숫자를 만드는 결정적 실험."""
from pathlib import Path
import json
import numpy as np

def compute():
    h=0.001; derivative=((3+h)**2-3**2)/h
    return {'finite_difference': round(derivative,3), 'exact': 6.0}

if __name__ == "__main__":
    result = compute()
    target = Path(__file__).with_name("results") / "ch05.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
