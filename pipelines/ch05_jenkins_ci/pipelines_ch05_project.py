"""5장 실습용 작은 프로젝트를 진짜 git 저장소로 만든다.

이 프로젝트는 3·4장의 고객 이탈 모델을 "여러 사람이 고치는 저장소" 모양으로 옮긴 것이다.
- churn/data.py : 4장(pipelines/ch04_container/pipelines_ch04_data.py)의 생성기에서 수치 6열 + 범주 2열 + 라벨을 복사.
- churn/model.py: 4장 pipelines_ch04_model_np.py의 로지스틱 회귀(전체 배치 경사하강법)를 복사.
- tests/      : 데이터 테스트와 성능 하한 테스트.
- ci/         : 이미지 지문 계산(4장 빌더를 아주 줄인 흉내)과 레지스트리 push 흉내.
- Jenkinsfile : 이 장 폴더의 Jenkinsfile을 그대로 넣는다.

장끼리 동시에 작성되므로 다른 장 모듈을 import하지 않고 문자열로 복사했다.
파일 내용(텍스트)을 dict로 들고 있고, make_repo()가 임시 폴더에 git 저장소를 만든다.
git 커밋 시각과 작성자는 고정해서 커밋 SHA가 매번 같게 한다.
"""

import os
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
FIXED_DATE = "2026-10-01T09:00:00+09:00"

DATA_PY = '''"""고객 이탈 합성 데이터(4장 생성기 복사)."""
import numpy as np

COLUMNS = ["tenure_months", "monthly_fee", "support_calls", "usage_hours", "age", "discount_rate"]
LABEL = "churned"


def make_table(seed, n_rows):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n_rows).astype(float)
    fee = np.round(np.clip(rng.normal(55_000, 15_000, n_rows), 15_000, 120_000), -2)
    calls = rng.poisson(np.clip(3.0 - tenure / 30.0, 0.2, None)).astype(float)
    usage = np.clip(rng.normal(40, 12, n_rows), 0, 200)
    age = rng.integers(19, 76, n_rows).astype(float)
    discount = rng.uniform(0.0, 0.3, n_rows)
    plan = rng.choice(3, n_rows, p=[0.5, 0.3, 0.2]).astype(np.int64)
    region = rng.choice(4, n_rows, p=[0.35, 0.30, 0.15, 0.20]).astype(np.int64)
    logit = (
        -0.3
        + 1.6 * (fee - 55_000) / 15_000
        - 1.3 * (tenure - 36) / 20
        + 0.5 * (calls - 1.8)
        - 0.8 * (usage - 40) / 12
        - 0.6 * (discount - 0.15) / 0.087
        + np.array([0.9, 0.0, -1.1])[plan]
        + np.array([0.0, 0.1, -0.1, 0.2])[region]
    )
    churned = (rng.uniform(size=n_rows) < 1.0 / (1.0 + np.exp(-logit))).astype(float)
    return {
        "tenure_months": tenure,
        "monthly_fee": fee,
        "support_calls": calls,
        "usage_hours": usage,
        "age": age,
        "discount_rate": discount,
        "plan": plan,
        "region": region,
        LABEL: churned,
    }
'''

FEATURES_PY = '''"""특성 행렬 만들기: 수치 열 표준화 + 범주 원-핫."""
import numpy as np

from churn.data import COLUMNS


def build_features(table, extra=()):
    cols = [(table[c] - table[c].mean()) / table[c].std() for c in COLUMNS]
    for f in extra:
        v = f(table)
        cols.append((v - v.mean()) / v.std())
    n = len(table[COLUMNS[0]])
    for k, levels in (("plan", 3), ("region", 4)):
        onehot = np.zeros((n, levels))
        onehot[np.arange(n), table[k]] = 1.0
        cols.extend(onehot.T)
    return np.column_stack(cols)
'''

MODEL_PY = '''"""로지스틱 회귀(4장 NumPy 구현 복사). 초기값 0, 고정 단계 수라 같은 입력이면 비트까지 같다."""
import numpy as np


def fit_logistic(X, y, lr=0.5, steps=400, l2=1e-3):
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        w -= lr * (X.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w, b


def predict_proba(X, w, b):
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))
'''

TRAIN_PY = '''"""작은 표본으로 학습하고 검증 정확도를 낸다. CI는 이걸로 성능 하한만 본다."""
import json
import sys
from pathlib import Path

import numpy as np

from churn.data import LABEL, make_table
from churn.features import build_features
from churn.model import fit_logistic, predict_proba

EXTRA = []


def run(data_seed=0, n_rows=1200, n_train=800, split_seed=1):
    table = make_table(data_seed, n_rows)
    X = build_features(table, EXTRA)
    y = table[LABEL]
    order = np.random.default_rng(split_seed).permutation(n_rows)
    tr, va = order[:n_train], order[n_train:]
    w, b = fit_logistic(X[tr], y[tr])
    acc = float(np.mean((predict_proba(X[va], w, b) >= 0.5) == y[va]))
    return {"val_acc": acc, "n_features": int(X.shape[1]), "w_sum": float(np.sum(w))}


if __name__ == "__main__":
    out = run()
    Path("results").mkdir(exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(out, sort_keys=True) + "\\n")
    print(json.dumps(out))
    sys.exit(0)
'''

TEST_DATA_PY = '''import numpy as np

from churn.data import COLUMNS, make_table


def test_columns_present():
    t = make_table(0, 200)
    assert all(c in t for c in COLUMNS)


def test_no_nan():
    t = make_table(0, 200)
    assert not any(np.isnan(t[c]).any() for c in COLUMNS)


def test_fee_range():
    t = make_table(0, 200)
    assert t["monthly_fee"].min() >= 15_000 and t["monthly_fee"].max() <= 120_000
'''

TEST_MODEL_PY = '''from churn.train import run

FLOOR = 0.75


def test_accuracy_floor():
    assert run()["val_acc"] >= FLOOR
'''

BUILD_IMAGE_PY = '''"""이미지 빌드 흉내: lock과 소스 파일 내용으로 지문을 만든다(4장 빌더의 원리만 남김).
실제 Docker 이미지를 만들지 않는다. 같은 내용이면 같은 지문, 한 바이트라도 다르면 다른 지문."""
import hashlib
import json
from pathlib import Path

files = ["requirements.lock"] + sorted(str(p) for p in Path("churn").glob("*.py"))
h = hashlib.sha256()
for f in files:
    h.update(f.encode() + b"\\0" + hashlib.sha256(Path(f).read_bytes()).hexdigest().encode() + b"\\n")
Path("results").mkdir(exist_ok=True)
out = {"digest": "sha256:" + h.hexdigest(), "files": files}
Path("results/image.json").write_text(json.dumps(out, sort_keys=True) + "\\n")
print(out["digest"])
'''

PUSH_PY = '''"""레지스트리 push 흉내: 커밋 SHA를 태그로 이미지 지문을 registry/ 폴더에 적는다."""
import json
import os
from pathlib import Path

sha = os.environ["GIT_COMMIT"]
img = json.loads(Path("results/image.json").read_text())
reg = Path(os.environ.get("REGISTRY_DIR", "registry"))
reg.mkdir(parents=True, exist_ok=True)
(reg / f"churn-{sha[:12]}.json").write_text(json.dumps({"tag": sha[:12], **img}, sort_keys=True) + "\\n")
print("pushed", sha[:12])
'''

BASE_FILES = {
    "churn/__init__.py": "",
    "churn/data.py": DATA_PY,
    "churn/features.py": FEATURES_PY,
    "churn/model.py": MODEL_PY,
    "churn/train.py": TRAIN_PY,
    "tests/test_data.py": TEST_DATA_PY,
    "tests/test_model.py": TEST_MODEL_PY,
    "ci/build_image.py": BUILD_IMAGE_PY,
    "ci/push.py": PUSH_PY,
    "requirements.lock": "numpy==2.3.3\npytest==8.4.2\n",
    "pytest.ini": "[pytest]\npythonpath = .\naddopts = -p no:cacheprovider\n",
    ".gitignore": "reports/\nresults/\nregistry/\n__pycache__/\n",
}


# ---------- 두 브랜치의 변경 ----------

def branch_a_rename(files):
    """브랜치 A: 요금 열 이름을 단위가 드러나게 바꾼다. A가 볼 수 있는 모든 사용처를 함께 고친다."""
    out = dict(files)
    out["churn/data.py"] = out["churn/data.py"].replace('"monthly_fee"', '"monthly_fee_krw"')
    out["tests/test_data.py"] = out["tests/test_data.py"].replace('t["monthly_fee"]', 't["monthly_fee_krw"]')
    return out


EXTRA_PY = '''"""새 특성: 상담 전화 한 번당 요금."""


def fee_per_call(table):
    return table["monthly_fee"] / (table["support_calls"] + 1.0)
'''

TEST_EXTRA_PY = '''from churn.data import make_table
from churn.extra import fee_per_call


def test_fee_per_call_positive():
    assert (fee_per_call(make_table(0, 200)) > 0).all()
'''


def branch_b_feature(files):
    """브랜치 B: 요금 열을 쓰는 새 특성을 다른 파일에 추가하고 학습에 넣는다. data.py는 건드리지 않는다."""
    out = dict(files)
    out["churn/extra.py"] = EXTRA_PY
    out["tests/test_extra.py"] = TEST_EXTRA_PY
    out["churn/train.py"] = out["churn/train.py"].replace(
        "from churn.model import fit_logistic, predict_proba\n\nEXTRA = []",
        "from churn.extra import fee_per_call\nfrom churn.model import fit_logistic, predict_proba\n\nEXTRA = [fee_per_call]",
    )
    return out


# ---------- git 도우미 ----------

def git_env(date=FIXED_DATE):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update({
        "GIT_AUTHOR_NAME": "lab", "GIT_AUTHOR_EMAIL": "lab@example.com",
        "GIT_COMMITTER_NAME": "lab", "GIT_COMMITTER_EMAIL": "lab@example.com",
        "GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date,
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
    })
    return env


def git(repo, *args, check=True, date=FIXED_DATE):
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env=git_env(date))
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} 실패: {r.stderr.strip()}")
    return r


def write_files(repo, files):
    repo = Path(repo)
    for rel, text in files.items():
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def commit_all(repo, message, date=FIXED_DATE):
    git(repo, "add", "-A", date=date)
    git(repo, "commit", "-q", "-m", message, date=date)
    return git(repo, "rev-parse", "HEAD").stdout.strip()


def jenkinsfile_text(name="Jenkinsfile"):
    return (HERE / name).read_text()


def make_repo(path, jenkinsfile=None):
    """main 브랜치에 기준 커밋 하나가 있는 저장소를 만든다. 돌려주는 값: 기준 커밋 SHA."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q", "-b", "main")
    files = dict(BASE_FILES)
    files["Jenkinsfile"] = jenkinsfile if jenkinsfile is not None else jenkinsfile_text()
    write_files(path, files)
    return commit_all(path, "base: churn model, data/model tests, Jenkinsfile")


def make_two_branches(path, jenkinsfile=None):
    """기준 커밋에서 A(열 이름 변경), B(새 특성) 브랜치를 만든다. 돌려주는 값: SHA dict."""
    base = make_repo(path, jenkinsfile)
    files = dict(BASE_FILES)
    files["Jenkinsfile"] = (Path(path) / "Jenkinsfile").read_text()
    git(path, "checkout", "-q", "-b", "branch-a")
    write_files(path, branch_a_rename(files))
    a = commit_all(path, "A: rename monthly_fee -> monthly_fee_krw", date="2026-10-02T10:00:00+09:00")
    git(path, "checkout", "-q", "main")
    git(path, "checkout", "-q", "-b", "branch-b")
    write_files(path, branch_b_feature(files))
    b = commit_all(path, "B: add fee_per_call feature", date="2026-10-02T11:00:00+09:00")
    git(path, "checkout", "-q", "main")
    return {"base": base, "a": a, "b": b}


def merge_both(path):
    """main에 A, 그다음 B를 합친다. 돌려주는 값: (합친 커밋 SHA, B를 합칠 때 텍스트 충돌 파일 목록)."""
    git(path, "merge", "-q", "--no-ff", "-m", "merge A", "branch-a", date="2026-10-03T09:00:00+09:00")
    r = git(path, "merge", "-q", "--no-ff", "-m", "merge B", "branch-b", check=False, date="2026-10-03T09:05:00+09:00")
    conflicts = [ln for ln in git(path, "diff", "--name-only", "--diff-filter=U").stdout.split() if ln]
    if r.returncode != 0 and not conflicts:
        raise RuntimeError(r.stderr)
    return git(path, "rev-parse", "HEAD").stdout.strip(), conflicts


def load_modules(tmpdir, files=None):
    """프로젝트 파일을 tmpdir에 쓰고 churn 패키지를 import해 돌려준다(실험·테스트에서 같은 코드를 직접 부르려고)."""
    import importlib
    import sys

    write_files(tmpdir, files or BASE_FILES)
    sys.path.insert(0, str(tmpdir))
    try:
        for name in [m for m in sys.modules if m == "churn" or m.startswith("churn.")]:
            del sys.modules[name]
        mods = {n: importlib.import_module(f"churn.{n}") for n in ("data", "features", "model", "train")}
    finally:
        sys.path.remove(str(tmpdir))
    return mods
