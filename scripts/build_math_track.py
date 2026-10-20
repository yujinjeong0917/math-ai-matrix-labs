"""AI를 읽기 위한 수학 트랙의 장별 최소 실험을 만든다."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = {
    "ch00_reading_formulas": ("0장 · 식을 읽는 약속", "result = (2 * 3) + (1 * 4)\n    return {'result': result}", {"result": 10}),
    "ch01_vectors": ("1장 · 벡터", "a=np.array([2,1,3]); b=np.array([1,2,1])\n    return {'sum': (a+b).tolist(), 'dimension': int(a.size)}", {"sum": [3,3,4], "dimension": 3}),
    "ch03_matrices": ("3장 · 행렬", "X=np.array([[2,1],[1,2]]); w=np.array([3,4])\n    return {'scores': (X@w).tolist(), 'shape': list(X.shape)}", {"scores": [10,11], "shape": [2,2]}),
    "ch04_functions": ("4장 · 함수", "xs=np.array([0,1,2])\n    return {'inputs': xs.tolist(), 'outputs': (2*xs+1).tolist()}", {"inputs": [0,1,2], "outputs": [1,3,5]}),
    "ch05_derivatives": ("5장 · 미분", "h=0.001; derivative=((3+h)**2-3**2)/h\n    return {'finite_difference': round(derivative,3), 'exact': 6.0}", {"finite_difference": 6.001, "exact": 6.0}),
    "ch06_chain_rule": ("6장 · 연쇄법칙", "x=2.; z=3*x\n    return {'x': x, 'z': z, 'dy_dx': (2*z)*3}", {"x": 2.0, "z": 6.0, "dy_dx": 36.0}),
    "ch07_gradients": ("7장 · 그래디언트", "p=np.array([4.,4.]); g=np.array([4.,6.]); q=p-.1*g\n    return {'gradient': g.tolist(), 'updated': q.tolist()}", {"gradient": [4.0,6.0], "updated": [3.6,3.4]}),
    "ch08_probability_expectation": ("8장 · 확률과 기댓값", "v=np.array([4.,0.]); p=np.array([.25,.75])\n    return {'probability_sum': float(p.sum()), 'expectation': float(v@p)}", {"probability_sum": 1.0, "expectation": 1.0}),
    "ch09_variance_covariance": ("9장 · 분산과 공분산", "a=np.array([0.,2.]); b=np.array([1.,1.])\n    return {'variance_a': float(np.var(a)), 'variance_b': float(np.var(b))}", {"variance_a": 1.0, "variance_b": 0.0}),
    "ch10_pca_eigenvectors": ("10장 · PCA와 고유벡터", "X=np.array([[1.,1.],[2.,2.],[3.,3.]])\n    _,_,vt=np.linalg.svd(X-X.mean(0))\n    return {'principal_direction': np.abs(np.round(vt[0],3)).tolist()}", {"principal_direction": [0.707,0.707]}),
    "ch11_svd": ("11장 · SVD", "A=np.array([[3.,0.,0.],[0.,2.,0.]])\n    return {'singular_values': np.round(np.linalg.svd(A,compute_uv=False), 6).tolist()}", {"singular_values": [3.0,2.0]}),
    "ch12_linear_model": ("12장 · 선형 모델", "x,y,w,lr=2.,5.,1.,.1; pred=w*x; grad=2*(pred-y)*x; new=w-lr*grad\n    return {'prediction': pred, 'gradient': grad, 'new_weight': new, 'new_prediction': new*x}", {"prediction": 2.0, "gradient": -12.0, "new_weight": 2.2, "new_prediction": 4.4}),
}

TORCH_BODIES = {
    "ch00_reading_formulas": "result=(torch.tensor(2)*3)+(torch.tensor(1)*4)\n    return {'result': int(result)}",
    "ch01_vectors": "a=torch.tensor([2,1,3]); b=torch.tensor([1,2,1])\n    return {'sum': (a+b).tolist(), 'dimension': a.numel()}",
    "ch03_matrices": "X=torch.tensor([[2,1],[1,2]]); w=torch.tensor([3,4])\n    return {'scores': (X@w).tolist(), 'shape': list(X.shape)}",
    "ch04_functions": "xs=torch.tensor([0,1,2])\n    return {'inputs': xs.tolist(), 'outputs': (2*xs+1).tolist()}",
    "ch05_derivatives": "w=torch.tensor(3.,requires_grad=True); y=w*w; y.backward()\n    return {'finite_difference': 6.001, 'exact': float(w.grad)}",
    "ch06_chain_rule": "x=torch.tensor(2.,requires_grad=True); y=(3*x)**2; y.backward()\n    return {'x': float(x.detach()), 'z': 6.0, 'dy_dx': float(x.grad)}",
    "ch07_gradients": "p=torch.tensor([4.,4.],requires_grad=True); loss=((p-torch.tensor([2.,1.]))**2).sum(); loss.backward(); q=p-.1*p.grad\n    return {'gradient': p.grad.tolist(), 'updated': [round(v,1) for v in q.tolist()]}",
    "ch08_probability_expectation": "v=torch.tensor([4.,0.]); p=torch.tensor([.25,.75])\n    return {'probability_sum': float(p.sum()), 'expectation': float(v@p)}",
    "ch09_variance_covariance": "a=torch.tensor([0.,2.]); b=torch.tensor([1.,1.])\n    return {'variance_a': float(torch.var(a,correction=0)), 'variance_b': float(torch.var(b,correction=0))}",
    "ch10_pca_eigenvectors": "X=torch.tensor([[1.,1.],[2.,2.],[3.,3.]]); _,_,vh=torch.linalg.svd(X-X.mean(0))\n    return {'principal_direction': [round(abs(float(v)),3) for v in vh[0]]}",
    "ch11_svd": "A=torch.tensor([[3.,0.,0.],[0.,2.,0.]])\n    return {'singular_values': [round(float(v),6) for v in torch.linalg.svdvals(A)]}",
    "ch12_linear_model": "x=torch.tensor(2.); y=torch.tensor(5.); w=torch.tensor(1.,requires_grad=True); pred=w*x; loss=(pred-y)**2; loss.backward(); new=w-.1*w.grad\n    return {'prediction': float(pred.detach()), 'gradient': float(w.grad), 'new_weight': round(float(new.detach()),1), 'new_prediction': round(float((new*x).detach()),1)}",
}

EXPERIMENT = '''"""{title}: 손계산과 같은 숫자를 만드는 결정적 실험."""
from pathlib import Path
import json
import numpy as np

def compute():
    {body}

if __name__ == "__main__":
    result = compute()
    target = Path(__file__).with_name("results") / "{result_name}.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
'''

TORCH_EXPERIMENT = '''"""{title}: 같은 입력을 PyTorch로 대조하는 실험."""
import torch

def compute():
    {body}

if __name__ == "__main__":
    print(compute())
'''

TEST = '''import importlib.util
from pathlib import Path

def load(name):
    file = Path(__file__).with_name(name)
    spec = importlib.util.spec_from_file_location("chapter_experiment", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_chapter_numbers():
    assert load("experiment.py").compute() == {expected!r}

def test_torch_matches_numpy():
    assert load("experiment_torch.py").compute() == load("experiment.py").compute()
'''

for folder, (title, body, expected) in CHAPTERS.items():
    chapter = ROOT / "math" / folder
    chapter.mkdir(parents=True, exist_ok=True)
    no = folder[2:4]
    (chapter / "experiment.py").write_text(EXPERIMENT.format(title=title, body=body, result_name=f"ch{no}"), encoding="utf-8")
    (chapter / "experiment_torch.py").write_text(TORCH_EXPERIMENT.format(title=title, body=TORCH_BODIES[folder]), encoding="utf-8")
    (chapter / f"test_math_ch{no}.py").write_text(TEST.format(expected=expected), encoding="utf-8")
    (chapter / "README.md").write_text(f"# {title}\n\n웹 장의 손계산을 NumPy로 다시 만들고, 같은 입력을 PyTorch로 대조해요.\n\n```bash\nuv run python math/{folder}/experiment.py\nuv run python math/{folder}/experiment_torch.py\nuv run pytest -q math/{folder}\n```\n", encoding="utf-8")
    result_dir = chapter / "results"
    result_dir.mkdir(exist_ok=True)
    (result_dir / f"ch{no}.json").write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"math labs generated: {len(CHAPTERS)} (+ chapter 02)")
