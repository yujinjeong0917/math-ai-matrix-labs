"""3장 실험 기록: 파일 하나(JSON Lines)와 결과 파일 폴더로 만든 가장 작은 실험 추적기.

기록 한 줄 = (run_id, 코드, 데이터 해시, 설정, 시드, 환경, 지표, 결과 파일 위치)
- log_run: 한 번의 실행을 runs.jsonl에 한 줄로 덧붙인다. 앞 줄은 고치지 않는다(덧붙이기만).
- load_runs: 기록을 모두 읽는다.
- rerun_from_record: 기록만 보고 같은 실행을 다시 해, 데이터 해시와 지표가 같은지 확인한다.
"""

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SOURCE_FILES = [
    "pipelines_ch03_data.py",
    "pipelines_ch03_model_np.py",
    "pipelines_ch03_model_torch.py",
    "pipelines_ch03_hashing.py",
    "pipelines_ch03_tracking.py",
    "experiments.py",  # 설정 묶음(GRID), 행 수, 학습·검증 분할 크기(N_TRAIN)가 여기 있다. 빼면 분할 크기를 바꿔도 코드 해시가 그대로다.
]


def code_identity():
    """git 커밋만으로는 '커밋 안 한 수정'을 구별하지 못하므로, 소스 파일 내용의 해시를 같이 남긴다."""
    h = hashlib.sha256()
    for name in SOURCE_FILES:
        h.update(name.encode() + b"\x00" + (HERE / name).read_bytes())
    commit, dirty = None, None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=HERE, capture_output=True, text=True, check=True).stdout
        dirty = bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"source_sha256": h.hexdigest(), "git_commit": commit, "git_dirty": dirty}


def env_snapshot():
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "machine": platform.machine(),
        "system": platform.system(),
    }


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def log_run(store, params, metrics, artifacts=None, data=None, code=None, env=None):
    """한 번의 실행을 기록하고 run_id를 돌려준다.

    params: 하이퍼파라미터와 시드(data_seed, split_seed 등). 다시 실행할 때 필요한 값 전부.
    data: {"train_valid": 데이터 해시, ...}
    artifacts: {이름: numpy 배열 딕셔너리}. 결과 파일은 store/artifacts/<run_id>/<이름>.npz로 저장한다.
    """
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    log = store / "runs.jsonl"
    index = sum(1 for _ in log.open()) if log.exists() else 0
    body = {"params": params, "data": data or {}, "code": code or {}, "env": env or {}}
    run_id = f"{index:04d}-{hashlib.sha256(_canonical(body).encode()).hexdigest()[:8]}"
    uris = {}
    for name, arrays in (artifacts or {}).items():
        path = store / "artifacts" / run_id / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        uris[name] = str(path.relative_to(store))
    record = {"run_id": run_id, **body, "metrics": metrics, "artifact_uri": uris}
    with log.open("a") as f:
        f.write(_canonical(record) + "\n")
    return run_id


def load_runs(store):
    log = Path(store) / "runs.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.open()]


def comparable_groups(runs, keys=("data", "code")):
    """같은 데이터 해시와 같은 코드 해시를 가진 기록끼리만 묶는다. 묶음 밖끼리 점수를 비교하지 않는다."""
    groups = {}
    for r in runs:
        key = (r["data"].get("train_valid"), r["code"].get("source_sha256"))
        groups.setdefault(key, []).append(r["run_id"])
    return groups


class RecordMismatch(RuntimeError):
    pass


def rerun_from_record(record, make_data, train_fn):
    """기록만으로 다시 실행한다. 데이터 해시가 기록과 다르면 학습하기 전에 멈춘다.

    make_data(params) -> (train, valid, data_hash), train_fn(train, valid, params) -> metrics
    """
    train, valid, h = make_data(record["params"])
    if h != record["data"].get("train_valid"):
        raise RecordMismatch(f"데이터 해시가 다르다: 기록 {record['data'].get('train_valid', '')[:12]}, 지금 {h[:12]}")
    return train_fn(train, valid, record["params"])
