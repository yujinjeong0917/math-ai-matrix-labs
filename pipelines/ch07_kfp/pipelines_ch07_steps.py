"""7장 단계 세 개: 데이터 준비(prepare) -> 학습(train) -> 평가(evaluate).

같은 함수를 두 방식으로 잇는다.
- run_all.sh: 파일 이름 규약으로 잇는다(이 장의 실패 예제). 명령줄로 부를 수 있게 맨 아래에 CLI가 있다.
- pipelines_ch07_dag.py: 입력·출력이 선언된 컴포넌트로 감싸 그래프로 잇는다(이 장의 전환).

각 단계는 입력 파일의 지문(sha256 앞 12자리)과, 데이터 파일 안에 적어 둔 만든 날(seed, shift)을 결과에 남긴다.
그래야 "어느 단계가 어느 입력을 썼는지"를 나중에 확인할 수 있다. run_all.sh 쪽에서도 이 기록은 남지만,
아무도 그걸 확인하지 않으면 틀린 연결은 조용히 지나간다는 게 이 장의 요점이다.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import pipelines_ch07_data as D
import pipelines_ch07_model_np as M

# prepare가 쓰는 파일 이름. 실패 예제 A에서는 누군가 prepare만 고쳐 이름을 바꾼 상황을
# 환경 변수 CH07_PREP_SUFFIX로 흉내 낸다(예: "_v2" -> train_v2.npz). run_all.sh는 옛 이름을 그대로 읽는다.
def _names():
    sfx = os.environ.get("CH07_PREP_SUFFIX", "")
    return f"train{sfx}.npz", f"valid{sfx}.npz"


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def _save_table(path, table, meta):
    """경로를 그대로 쓴다. np.savez(path)에 문자열을 주면 '.npz'를 덧붙여서, KFP가 정해 준 경로(.path)와
    다른 파일이 생기고 다음 단계가 못 찾는다(kfp.local로 실제로 돌려 보다 겪은 실패). 그래서 파일 핸들로 쓴다."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, _meta=np.array(json.dumps(meta)), **table)


def load_table(path):
    with np.load(path) as z:
        meta = json.loads(str(z["_meta"]))
        table = {k: z[k] for k in z.files if k != "_meta"}
    return table, meta


def prepare(seed, n_rows, shift, train_path, valid_path):
    """오늘의 표를 만들고 학습 3/4, 검증 1/4로 나눠 저장한다. CH07_PREP_FAIL=1이면 아무것도 쓰기 전에 실패한다."""
    if os.environ.get("CH07_PREP_FAIL") == "1":
        raise RuntimeError("prepare: 원천 테이블을 읽다가 실패했다(흉내)")
    t = D.make_table(seed, n_rows, shift=shift)
    tr, va = D.split(t, n_rows * 3 // 4, seed)
    meta = {"seed": seed, "shift": shift, "n_rows": n_rows}
    _save_table(train_path, tr, dict(meta, part="train"))
    _save_table(valid_path, va, dict(meta, part="valid"))
    return {"train_rows": len(tr[D.LABEL]), "valid_rows": len(va[D.LABEL])}


def train(data_path, model_path, impl="numpy", steps=400):
    table, meta = load_table(data_path)
    if impl == "torch":
        import pipelines_ch07_model_torch as MT
        model = M.train(table, fit=MT.fit_logistic, steps=steps)
    else:
        model = M.train(table, steps=steps)
    model["trained_on"] = {"sha": fingerprint(data_path), "seed": meta["seed"], "shift": meta["shift"]}
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_path).write_text(json.dumps(model))
    return model["trained_on"]


def evaluate(model_path, data_path, metrics_path, threshold=0.5):
    model = json.loads(Path(model_path).read_text())
    table, meta = load_table(data_path)
    m = M.evaluate(model, table, threshold)
    m["evaluated_on"] = {"sha": fingerprint(data_path), "seed": meta["seed"], "shift": meta["shift"]}
    m["trained_on"] = model["trained_on"]
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(m))
    return m


def _cli():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-rows", type=int, default=4000)
    p.add_argument("--shift", type=float, default=0.0)
    p.add_argument("--outdir", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--impl", default="numpy")
    e = sub.add_parser("evaluate")
    e.add_argument("--model", required=True)
    e.add_argument("--data", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--threshold", type=float, default=0.5)
    a = ap.parse_args()
    if a.cmd == "prepare":
        tn, vn = _names()
        print("prepare", prepare(a.seed, a.n_rows, a.shift, Path(a.outdir) / tn, Path(a.outdir) / vn))
    elif a.cmd == "train":
        print("train", train(a.data, a.out, a.impl))
    else:
        print("evaluate", evaluate(a.model, a.data, a.out, a.threshold))


if __name__ == "__main__":
    _cli()
