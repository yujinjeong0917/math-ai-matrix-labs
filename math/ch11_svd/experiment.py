"""11장 · SVD: 손계산과 같은 숫자를 만드는 결정적 실험."""
from pathlib import Path
import json
import numpy as np

def compute():
    A=np.array([[3.,0.,0.],[0.,2.,0.]])
    return {'singular_values': np.round(np.linalg.svd(A,compute_uv=False), 6).tolist()}

if __name__ == "__main__":
    result = compute()
    target = Path(__file__).with_name("results") / "ch11.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
