"""7장 실제 KFP 확인(선택). 이 저장소의 uv 환경에는 kfp가 없으므로 따로 만든 환경에서 돌린다.

    uv venv -p 3.12 /tmp/kfp && uv pip install -p /tmp/kfp/bin/python kfp==2.17.0 numpy==2.3.3 pyyaml
    PYTHONPATH=pipelines/ch07_kfp OMP_NUM_THREADS=2 /tmp/kfp/bin/python pipelines/ch07_kfp/kfp_local_check.py

하는 일
1. kfp_pipeline.py를 IR YAML로 컴파일해 results/kfp_ir_2.17.0.yaml로 저장하고, 같은 내용을 JSON으로도 저장한다
   (실습 저장소 테스트는 pyyaml 없이 JSON만 읽는다).
2. kfp.local.SubprocessRunner(use_venv=False)로 같은 인자의 실행을 3번 하고, 걸린 시간과 결과 파일 위치를 기록한다.
   클러스터가 아니라 내 컴퓨터의 하위 프로세스에서 도는 것이다. 컨테이너 기동 시간은 들어 있지 않다.
3. 출력 이름을 바꾼 컴포넌트를 옛 이름으로 이을 때, 타입이 다른 입력에 이을 때 KFP가 언제 무엇을 내는지 기록한다.
4. prepare가 실패하면 뒤 단계가 도는지 기록한다.
결과는 results/ch07_kfp_local.json. experiments.py가 이 파일이 있으면 ch07.json에 합친다.
"""

import inspect
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import yaml
import kfp
from kfp import compiler, dsl, local
from kfp.dsl import Dataset, Input, Output

import kfp_pipeline as K

HERE = Path(__file__).parent
RES = HERE / "results"


def compile_ir():
    y = RES / f"kfp_ir_{kfp.__version__}.yaml"
    compiler.Compiler().compile(K.churn_pipeline, package_path=str(y))
    spec = yaml.safe_load(y.read_text())
    (RES / f"kfp_ir_{kfp.__version__}.json").write_text(json.dumps(spec, indent=1, sort_keys=True))
    return y.name


def wiring_errors():
    @dsl.component(base_image="python:3.12")
    def prep_renamed(train_v2: Output[Dataset]):
        pass

    @dsl.component(base_image="python:3.12")
    def needs_dataset(train_data: Input[Dataset]):
        pass

    @dsl.component(base_image="python:3.12")
    def needs_int(train_data: int):
        pass

    out = {}
    for case in ("renamed_output", "type_mismatch"):
        stage = "definition"
        try:
            @dsl.pipeline
            def p():
                a = prep_renamed()
                if case == "renamed_output":
                    needs_dataset(train_data=a.outputs["train_data"])
                else:
                    needs_int(train_data=a.outputs["train_v2"])
            stage = "compile"
            compiler.Compiler().compile(p, package_path=os.devnull)
            out[case] = {"raised": None}
        except Exception as e:  # noqa: BLE001
            out[case] = {"raised": type(e).__name__, "stage": stage, "message": _plain(str(e))[:200]}
    return out


def run_once(root):
    t0 = time.perf_counter()
    K.churn_pipeline(seed=1, n_rows=4000, shift=1.0)
    sec = time.perf_counter() - t0
    run_dir = sorted(Path(root).iterdir(), key=lambda p: p.stat().st_mtime)[-1]
    metrics = json.loads((run_dir / "evaluate" / "metrics.json").read_text())
    files = sorted(str(p.relative_to(run_dir)) for p in run_dir.rglob("*") if p.is_file() and p.name != "executor_output.json")
    return {"seconds": round(sec, 3), "run_dir": run_dir.name, "files": files,
            "accuracy": metrics["accuracy"], "trained_on": metrics["trained_on"], "evaluated_on": metrics["evaluated_on"]}


def _plain(msg):
    return re.sub(r"\x1b\[[0-9;]*m", "", msg)


def client_signatures():
    return {name: [p for p in inspect.signature(getattr(kfp.Client, name)).parameters if p != "self"]
            for name in ("create_run_from_pipeline_package", "wait_for_run_completion")}


def main():
    RES.mkdir(exist_ok=True)
    res = {"kfp": kfp.__version__, "python": sys.version.split()[0], "ir_file": compile_ir(), "wiring": wiring_errors(),
           "client_signatures": client_signatures()}
    root = Path(tempfile.mkdtemp(prefix="kfp-ch07-"))
    try:
        local.init(runner=local.SubprocessRunner(use_venv=False), pipeline_root=str(root))
        res["runs"] = [run_once(root) for _ in range(3)]
        os.environ["CH07_PREP_FAIL"] = "1"
        try:
            K.churn_pipeline(seed=1, n_rows=4000, shift=1.0)
            res["prepare_failure"] = {"raised": None}
        except Exception as e:  # noqa: BLE001
            run_dir = sorted(root.iterdir(), key=lambda p: p.stat().st_mtime)[-1]
            res["prepare_failure"] = {"raised": type(e).__name__, "message": _plain(str(e))[:160],
                                      "task_dirs": sorted(p.name for p in run_dir.iterdir() if p.is_dir())}
        finally:
            os.environ.pop("CH07_PREP_FAIL")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    (RES / "ch07_kfp_local.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
