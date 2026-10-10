"""8장 단계 세 개: prepare -> train -> evaluate. 7장 pipelines_ch07_steps.py를 고쳐 복사했다.

7장과 다른 점
1. 모델 파일에는 가중치와 전처리 값만 쓴다. "어느 데이터로 학습했나"는 파일이 아니라 메타데이터 저장소
   (pipelines_ch08_store.py)에 남긴다. 배포되는 건 보통 모델 파일 하나뿐이라는 실패 예제 A를 위해서다.
2. prepare가 쓰는 데이터 규칙(source_version)을 세 곳 중 하나에서 정한다. 우선순위는
   파이프라인 파라미터 data_version > 환경 변수 SOURCE_VERSION > 이미지 안에 들어 있는 기본값.
   이미지 안의 기본값은 runner가 레지스트리에서 꺼낸 이미지 내용(ctx["image"])으로 흉내 낸다.
3. 각 단계는 결과 파일의 메타데이터(dict)를 돌려준다. runner가 이것을 저장소의 artifact 속성으로 적는다.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

import pipelines_ch08_data as D
import pipelines_ch08_model_np as M


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _save_table(path, table, meta):
    """7장에서 겪은 실패 그대로: np.savez에 문자열 경로를 주면 '.npz'가 붙는다. 파일 핸들로 쓴다."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, _meta=np.array(json.dumps(meta)), **table)


def load_table(path):
    with np.load(path) as z:
        meta = json.loads(str(z["_meta"]))
        table = {k: z[k] for k in z.files if k != "_meta"}
    return table, meta


def resolve_source_version(data_version, ctx):
    if data_version:
        return data_version, "parameter"
    env = (ctx or {}).get("env", {})
    if env.get("SOURCE_VERSION"):
        return env["SOURCE_VERSION"], "env"
    return (ctx or {}).get("image", {}).get("source_version_default", "v1"), "image_default"


def prepare(seed, n_rows, shift, train_path, valid_path, data_version="", ctx=None):
    version, decided_by = resolve_source_version(data_version, ctx)
    t = D.make_table(seed, n_rows, shift=shift, source_version=version)
    n = len(t[D.LABEL])
    tr, va = D.split(t, n * 3 // 4, seed)
    meta = {"seed": seed, "shift": shift, "n_rows": n_rows, "source_version": version, "source_version_from": decided_by}
    _save_table(train_path, tr, dict(meta, part="train"))
    _save_table(valid_path, va, dict(meta, part="valid"))
    return {
        "train_data": dict(meta, part="train", rows=len(tr[D.LABEL]), dataset_hash=D.dataset_hash(tr)),
        "valid_data": dict(meta, part="valid", rows=len(va[D.LABEL]), dataset_hash=D.dataset_hash(va)),
    }


def train(train_path, model_path, impl="numpy", steps=400, ctx=None):
    table, _ = load_table(train_path)
    if impl == "torch":
        import pipelines_ch08_model_torch as MT
        model = M.train(table, fit=MT.fit_logistic, steps=steps)
    elif impl == "numpy":
        model = M.train(table, steps=steps)
    else:
        raise ValueError(impl)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_path).write_text(json.dumps(model, sort_keys=True))
    return {"model": {"impl": impl, "steps": steps}}


def evaluate(model_path, valid_path, metrics_path, threshold=0.5, ctx=None):
    model = json.loads(Path(model_path).read_text())
    table, _ = load_table(valid_path)
    m = M.evaluate(model, table, threshold)
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metrics_path).write_text(json.dumps(m, sort_keys=True))
    return {"metrics": {k: round(float(v), 4) for k, v in m.items()}}


def load_model_weights(model_path):
    """배포된 모델 파일에서 읽을 수 있는 전부."""
    m = json.loads(Path(model_path).read_text())
    return np.array(m["w"]), m["b"], m["prep"]
