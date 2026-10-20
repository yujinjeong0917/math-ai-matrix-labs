"""0장 · 식을 읽는 약속: 손계산과 같은 숫자를 만드는 결정적 실험."""
from pathlib import Path
import json
import numpy as np

def compute():
    result = (2 * 3) + (1 * 4)
    return {'result': result}

if __name__ == "__main__":
    result = compute()
    target = Path(__file__).with_name("results") / "ch00.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
