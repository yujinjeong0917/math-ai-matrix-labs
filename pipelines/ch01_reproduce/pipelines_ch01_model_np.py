"""1장 모델: 무작위가 세 군데 들어가는 작은 학습 스크립트(NumPy).

무작위가 들어가는 곳
1. 분할: 2,000행을 8:2로 나눌 때 어느 행이 시험 쪽에 가나
2. 초기값: 가중치를 작은 정규분포 값으로 시작한다
3. 순서: 미니배치 경사하강법에서 매 에폭 행 순서를 섞는다

이 저장소는 numpy·torch만 쓰므로 scikit-learn 대신 미니배치 로지스틱 회귀를 직접 짰다.
시드를 안 주면 위 세 곳이 실행마다 바뀌어 결과가 달라진다는 성질은 같다.
"""

import hashlib

import numpy as np

from pipelines_ch01_data import make_dataset

DEFAULT_CFG = {
    "seed": 0,  # None이면 운영체제에서 새 엔트로피를 받는다(시드를 안 준 실행)
    "split_seed": None,  # 주면 분할만 이 시드로(분해 실험용)
    "train_seed": None,  # 주면 초기값·순서만 이 시드로(분해 실험용)
    "entropy": None,  # 시드 없이 돌린 실행이 받은 엔트로피를 기록해 두었다면, 그걸로 다시 낼 수 있다
    "test_frac": 0.2,
    "epochs": 5,
    "batch": 32,
    "lr": 0.1,
    "init_scale": 0.1,
    "dtype": "float64",
    "schema_version": 1,
}


def seed_streams(cfg):
    """분할용, 학습용 난수 생성기 두 개와 실제로 쓴 엔트로피를 돌려준다."""
    if cfg.get("entropy") is not None:
        root = np.random.SeedSequence(cfg["entropy"])
    elif cfg.get("seed") is None:
        root = np.random.SeedSequence()  # 운영체제에서 새로 받은 엔트로피. 기록하지 않으면 다시 못 낸다
    else:
        root = np.random.SeedSequence(cfg["seed"])
    split_ss, train_ss = root.spawn(2)
    if cfg.get("split_seed") is not None:
        split_ss = np.random.SeedSequence(cfg["split_seed"])
    if cfg.get("train_seed") is not None:
        train_ss = np.random.SeedSequence(cfg["train_seed"])
    return np.random.default_rng(split_ss), np.random.default_rng(train_ss), root.entropy


def split(n, test_frac, rng):
    idx = rng.permutation(n)
    n_test = int(round(n * test_frac))
    return idx[n_test:], idx[:n_test]


def standardize(X_tr, X_te, n_numeric=6):
    """수치 열만 학습 분할의 평균·표준편차로 표준화한다."""
    m = X_tr[:, :n_numeric].mean(axis=0)
    s = X_tr[:, :n_numeric].std(axis=0)
    out = []
    for X in (X_tr, X_te):
        X = X.copy()
        X[:, :n_numeric] = (X[:, :n_numeric] - m) / s
        out.append(X)
    return out


def batch_orders(n, epochs, batch, rng):
    """에폭마다 섞은 행 순서를 미리 뽑아 둔다. PyTorch 대조에서도 같은 순서를 쓴다."""
    return [rng.permutation(n) for _ in range(epochs)]


def fit_sgd(X, y, w0, b0, orders, batch, lr):
    """미니배치 경사하강법. 같은 (X, y, 초기값, 순서)면 결과도 같다."""
    w, b = w0.copy(), b0
    for order in orders:
        for start in range(0, len(order), batch):
            j = order[start:start + batch]
            p = 1.0 / (1.0 + np.exp(-(X[j] @ w + b)))
            g = p - y[j]
            w -= lr * (X[j].T @ g) / len(j)
            b -= lr * g.mean()
    return w, b


def predict_proba(X, w, b):
    return 1.0 / (1.0 + np.exp(-(X @ w + b)))


def fingerprint(*arrays):
    """배열의 바이트를 sha256으로. 다른 컴퓨터의 결과와 비트 단위로 비교할 때 쓴다."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def prepare(cfg):
    """분할·표준화·초기값·순서까지. 학습 직전 상태를 돌려준다(PyTorch 대조용)."""
    c = {**DEFAULT_CFG, **cfg}
    dt = np.dtype(c["dtype"])
    X, y = make_dataset(schema_version=c["schema_version"])
    split_rng, train_rng, entropy = seed_streams(c)
    tr, te = split(len(y), c["test_frac"], split_rng)
    X_tr, X_te = standardize(X[tr], X[te])
    X_tr, X_te, y_tr, y_te = X_tr.astype(dt), X_te.astype(dt), y[tr].astype(dt), y[te].astype(dt)
    w0 = (train_rng.normal(0.0, c["init_scale"], X.shape[1])).astype(dt)
    b0 = dt.type(0.0)
    orders = batch_orders(len(y_tr), c["epochs"], c["batch"], train_rng)
    return {"cfg": c, "X_tr": X_tr, "y_tr": y_tr, "X_te": X_te, "y_te": y_te, "w0": w0, "b0": b0,
            "orders": orders, "test_idx": te, "entropy": entropy}


def train(cfg):
    """cfg -> metrics. 시드 없이 부르면 실행마다 다른 값이 나온다."""
    s = prepare(cfg)
    c = s["cfg"]
    w, b = fit_sgd(s["X_tr"], s["y_tr"], s["w0"], s["b0"], s["orders"], c["batch"], c["lr"])
    p = predict_proba(s["X_te"], w, b)
    pred = (p >= 0.5).astype(float)
    correct = int(np.sum(pred == s["y_te"]))
    return {
        "accuracy": correct / len(s["y_te"]),
        "correct": correct,
        "n_test": len(s["y_te"]),
        "entropy": str(s["entropy"]),
        "test_idx_head": s["test_idx"][:5].tolist(),
        "weights_fp": fingerprint(w, np.asarray(b)),
        "proba_fp": fingerprint(p),
        "_w": w,
        "_b": float(b),
        "_p": p,
    }
