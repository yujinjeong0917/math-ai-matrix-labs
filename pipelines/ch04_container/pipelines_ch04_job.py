"""4장 작업 스크립트: 어느 환경에서 돌려도 같은 일을 하고, 결과를 JSON 한 줄로 찍는다.

experiments.py가 서로 다른 NumPy 버전 환경(uv --isolated)에서 이 파일을 실행해 결과를 비교한다.
NumPy만 쓴다(1.26에서도 돌아가야 하므로). 3장과 같은 데이터·설정으로 학습한다.

  python pipelines_ch04_job.py                 # 학습하고 지표·지문을 출력
  python pipelines_ch04_job.py --save DIR      # 모델을 pickle과 npz 두 형식으로 저장
  python pipelines_ch04_job.py --load DIR      # 두 형식을 각각 불러 보고 성공 여부를 출력
"""

import hashlib
import json
import pickle
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402

import pipelines_ch04_data as dat  # noqa: E402
import pipelines_ch04_model_np as mnp  # noqa: E402

DATA_SEED, N_ROWS, N_TRAIN, SPLIT_SEED = 100, 3000, 2000, 7
LR, STEPS, L2 = 0.5, 400, 0.01
N_SERVE = 100_000


def table_hash(table):
    """3장 dataset_hash와 같은 규칙(열 이름 정렬, 타입·모양·바이트)."""
    h = hashlib.sha256()
    for name in sorted(table):
        a = np.ascontiguousarray(table[name])
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        h.update(name.encode())
        h.update(b"\x00" + a.dtype.str.encode() + b"\x00" + repr(a.shape).encode() + b"\x00")
        h.update(a.tobytes())
    return h.hexdigest()


def train():
    table = dat.make_table(DATA_SEED, N_ROWS)
    tr, va = dat.split(table, N_TRAIN, SPLIT_SEED)
    model, metrics = mnp.train_and_eval(tr, va, LR, STEPS, L2)
    return table, tr, model, metrics


def serve_probe(tr, model):
    """float32로 들어온 서빙 입력 10만 행을 np.float64 통계로 변환해 확률을 낸다."""
    stats = mnp.serving_stats(tr)
    serve = dat.make_table(DATA_SEED + 1, N_SERVE)
    serve32 = {k: (v.astype(np.float32) if k in dat.NUMERIC else v) for k, v in serve.items()}
    dtype, Xnum = mnp.transform_serving(serve32, stats)
    k = len(dat.NUMERIC)
    p = 1.0 / (1.0 + np.exp(-(Xnum @ model["w"][:k] + model["b"])))  # 수치 열만 쓰는 점수(타입 차이만 보려고)
    return dtype, p


def main(argv):
    table, tr, model, metrics = train()
    out = {
        "python": platform.python_version(), "numpy": np.__version__,
        "data_hash": table_hash(table),
        "col_hashes": {k: table_hash({k: v})[:12] for k, v in table.items()},
        "val_acc": metrics["val_acc"],
        "w": [float(x) for x in model["w"]], "b": float(model["b"]),
        "w_sha256": hashlib.sha256(np.asarray(model["w"], dtype="<f8").tobytes()).hexdigest(),
        "nep50_scalar": str((np.float32(3) + 3.0).dtype),          # NEP 50 표의 예
        "nep50_array": str((np.array([1.0], np.float32) + np.float64(1.0)).dtype),
    }
    dtype, p = serve_probe(tr, model)
    out["serve_dtype"] = dtype
    out["serve_p_sha256"] = hashlib.sha256(p.astype("<f8").tobytes()).hexdigest()
    out["serve_p_head"] = [float(x) for x in p[:3]]
    if "--dump-p" in argv:
        np.save(argv[argv.index("--dump-p") + 1], p.astype("<f8"))
    rng = np.random.default_rng(1)
    a = rng.normal(40, 12, 3000)
    z = np.random.default_rng(1).standard_normal(3000)
    out["normal_eq_affine_frac"] = float(np.mean(a == 40 + 12 * z))  # normal(40,12) == 40 + 12*z 인 비율
    if "--train-on" in argv:  # 다른 환경이 만든 표로 학습해, 차이가 데이터 때문인지 계산 때문인지 가른다
        t = dict(np.load(argv[argv.index("--train-on") + 1]))
        tr2, va2 = dat.split(t, N_TRAIN, SPLIT_SEED)
        m2, s2 = mnp.train_and_eval(tr2, va2, LR, STEPS, L2)
        out["w_sha256_on_given"] = hashlib.sha256(np.asarray(m2["w"], dtype="<f8").tobytes()).hexdigest()
        out["val_acc_on_given"] = s2["val_acc"]
    if "--dump-table" in argv:
        np.savez(argv[argv.index("--dump-table") + 1], **table)
    if "--save" in argv:
        d = Path(argv[argv.index("--save") + 1])
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "model.pkl", "wb") as f:
            pickle.dump({"w": model["w"], "b": np.float64(model["b"])}, f)
        np.savez(d / "model.npz", w=model["w"], b=np.array(model["b"]))
    if "--load" in argv:
        d = Path(argv[argv.index("--load") + 1])
        loads = {}
        for fmt in ("pkl", "npz"):
            try:
                if fmt == "pkl":
                    with open(d / "model.pkl", "rb") as f:
                        m = pickle.load(f)
                    w = m["w"]
                else:
                    w = np.load(d / "model.npz")["w"]
                loads[fmt] = {"ok": True, "same_w": bool(np.array_equal(w, model["w"]))}
            except Exception as e:  # 어떤 오류가 나는지 그대로 기록한다
                loads[fmt] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        out["load"] = loads
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1:])
