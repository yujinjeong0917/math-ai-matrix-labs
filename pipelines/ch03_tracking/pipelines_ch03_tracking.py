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
# 소스 해시에 포함할 파일 목록. 목록에 없는 파일을 고치면 해시가 변하지 않는다
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
        # git이 없거나 저장소가 아니면 commit, dirty를 None으로 둔다
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
        # 이 폴더(.)에 커밋되지 않은 변경이 있으면 dirty=True
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
    # 키 정렬과 공백 제거로 같은 내용이 항상 같은 문자열이 되게 한다(해시 입력과 jsonl 한 줄 양쪽에 쓴다)
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
    # run_id 앞 번호 = 지금까지 기록된 줄 수. 동시에 두 프로세스가 쓰면 번호가 겹칠 수 있다(동시 실행은 고려하지 않은 단순화)
    index = sum(1 for _ in log.open()) if log.exists() else 0
    body = {"params": params, "data": data or {}, "code": code or {}, "env": env or {}}
    # run_id = 순번 4자리 + 설정·데이터·코드·환경 내용 해시 앞 8자리
    run_id = f"{index:04d}-{hashlib.sha256(_canonical(body).encode()).hexdigest()[:8]}"
    uris = {}
    for name, arrays in (artifacts or {}).items():
        path = store / "artifacts" / run_id / f"{name}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        # 결과 파일(가중치 등)은 jsonl에 넣지 않고 .npz로 따로 저장하고 상대 경로만 기록한다
        np.savez(path, **arrays)
        uris[name] = str(path.relative_to(store))
    record = {"run_id": run_id, **body, "metrics": metrics, "artifact_uri": uris}
    # append 모드: 앞 줄은 수정하지 않고 한 줄만 덧붙인다
    with log.open("a") as f:
        f.write(_canonical(record) + "\n")
    return run_id


def load_runs(store):
    log = Path(store) / "runs.jsonl"
    if not log.exists():
        return []
    return [json.loads(line) for line in log.open()]


# keys 인자는 함수 안에서 쓰이지 않는다. 실제 묶음 기준은 (데이터 해시, 소스 해시)로 고정
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
        # 데이터가 기록과 다르면 학습 전에 중단한다. 다른 데이터에서 나온 점수를 같은 실행으로 착각하지 않게 하는 장치
        raise RecordMismatch(f"데이터 해시가 다르다: 기록 {record['data'].get('train_valid', '')[:12]}, 지금 {h[:12]}")
    return train_fn(train, valid, record["params"])
