"""7장 실패 예제를 돌리는 도우미: run_all.sh를 진짜 bash로 돌리고, 두 실행이 한 폴더를 같이 쓰는 경우를 순서대로 흉내 낸다."""

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pipelines_ch07_steps as S

HERE = Path(__file__).parent
SCRIPT = HERE / "run_all.sh"


def run_glue(workdir, seed, shift, strict="", extra_env=None):
    """bash run_all.sh를 실제로 실행한다. (종료 코드, metrics.json 내용 또는 None)."""
    env = {**os.environ, "PY": sys.executable, "CH07_STRICT": strict, "OMP_NUM_THREADS": "2"}
    env.update(extra_env or {})
    r = subprocess.run(["bash", str(SCRIPT), str(workdir), str(seed), str(shift)], env=env, capture_output=True, text=True)
    mpath = Path(workdir) / "metrics.json"
    metrics = json.loads(mpath.read_text()) if mpath.exists() else None
    return {"exit_code": r.returncode, "metrics": metrics, "stderr_tail": r.stderr.strip().splitlines()[-1:] if r.stderr.strip() else []}


def interleavings(n_a=3, n_b=3):
    """두 실행의 단계(각 3개)를 각자의 순서는 지키며 섞는 모든 방법. C(6,3) = 20가지."""
    for pos in itertools.combinations(range(n_a + n_b), n_a):
        seq, ia, ib = [], 0, 0
        for k in range(n_a + n_b):
            if k in pos:
                seq.append(("A", ia)); ia += 1
            else:
                seq.append(("B", ib)); ib += 1
        yield seq


def shared_folder_steps(workdir, run, seed, shift):
    """run_all.sh와 같은 파일 이름 규약으로 세 단계를 함수로 돌린다(한 폴더를 두 실행이 같이 쓰는 상황용)."""
    w = Path(workdir)
    return [
        lambda: S.prepare(seed, 4000, shift, w / "train.npz", w / "valid.npz"),
        lambda: S.train(w / "train.npz", w / "model.json"),
        lambda: S.evaluate(w / "model.json", w / "valid.npz", w / f"metrics_{run}.json"),
    ]
