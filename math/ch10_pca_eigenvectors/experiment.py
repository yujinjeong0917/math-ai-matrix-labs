"""10장 · PCA와 고유벡터: 손계산과 같은 숫자를 만드는 결정적 실험."""
from pathlib import Path
import json
import numpy as np

def compute():
    X=np.array([[1.,1.],[2.,2.],[3.,3.]])
    _,_,vt=np.linalg.svd(X-X.mean(0))
    return {'principal_direction': np.abs(np.round(vt[0],3)).tolist()}

if __name__ == "__main__":
    result = compute()
    target = Path(__file__).with_name("results") / "ch10.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
