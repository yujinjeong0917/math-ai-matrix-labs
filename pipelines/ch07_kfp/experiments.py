"""7장 비교 실험. `uv run python pipelines/ch07_kfp/experiments.py` 로 실행한다(네트워크·Jenkins·쿠버네티스·kfp 불필요).

E1 파일 이름으로 이은 스크립트(run_all.sh, 진짜 bash) vs 그래프로 이은 파이프라인(pipelines_ch07_dag):
   A 출력 이름이 바뀜  B 앞 단계 실패(set -e 없음 / set -e + tee / set -eo pipefail)  C 두 실행이 한 폴더를 같이 씀(20가지 순서)
E2 같은 인자로 3번: 지표·지문이 같은지, 결과 파일 위치는 실행마다 다른지
E3 시간: 스크립트, 같은 프로세스에서 task 실행, task마다 새 프로세스(컨테이너 대신). 행 20만 개로 크게 잼
E4 한 단계만 바꿨을 때(평가 문턱 0.5 -> 0.4) 캐시 없이 전체를 다시 돌리는 시간(8장 캐시 비교의 기준선)
E5 train 컴포넌트 몸통을 NumPy -> PyTorch로 바꿔 끼우기: 그래프는 그대로, 지표는 같은지
E6 이미지 태그: ':latest'로 제출하면 시험한 이미지와 다른 이미지로 도는 run이 얼마나 되나(커밋 간격 120·30·10분)
E7 Jenkinsfile 규칙 검사: 지금 파일과 일부러 망가뜨린 세 가지
IR 실제 kfp 2.17.0 컴파일 결과(results/kfp_ir_2.17.0.json)와 우리 최소 구현의 그래프 비교
KFP results/ch07_kfp_local.json(kfp_local_check.py가 kfp 2.17.0 환경에서 만든 것)이 있으면 그대로 합친다.

결과는 results/ch07.json 에 저장하고, 웹 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pipelines_ch07_dag as G  # noqa: E402
import pipelines_ch07_glue as GL  # noqa: E402
import pipelines_ch07_jenkinsfile as JF  # noqa: E402
import pipelines_ch07_pipeline as P  # noqa: E402
import pipelines_ch07_registry as R  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
RES = HERE / "results"
OUT = RES / "ch07.json"
YESTERDAY = {"seed": 0, "shift": 0.0}
TODAY = {"seed": 1, "shift": 1.0}


def r4(x):
    return round(float(x), 4)


def brief(m):
    if m is None:
        return None
    return {"accuracy": r4(m["accuracy"]), "churn_rate": r4(m["churn_rate"]), "predicted_churn_rate": r4(m["predicted_churn_rate"]),
            "trained_on_seed": m["trained_on"]["seed"], "evaluated_on_seed": m["evaluated_on"]["seed"]}


def _metrics_of(run):
    ev = [t for t in run["tasks"] if t["task"] == "evaluate"][0]
    if ev["status"] != "SUCCEEDED":
        return None
    return json.loads(Path(ev["outputs"]["metrics"]["uri"] + ".json").read_text())


def e1(tmp):
    ir = G.compile(P.churn_pipeline)
    out = {}
    # 기준: 오늘 데이터를 제대로 이었을 때
    truth = GL.run_glue(tmp / "clean", **TODAY)
    out["today_correct"] = brief(truth["metrics"])
    yday = GL.run_glue(tmp / "y_only", **YESTERDAY)
    out["yesterday"] = brief(yday["metrics"])

    # A: prepare의 출력 이름이 train_v2.npz / valid_v2.npz로 바뀜. 스크립트는 옛 이름을 읽는다
    w = tmp / "A"
    GL.run_glue(w, **YESTERDAY)
    a = GL.run_glue(w, **TODAY, extra_env={"CH07_PREP_SUFFIX": "_v2"})
    out["A_glue"] = {"exit_code": a["exit_code"], "reported": brief(a["metrics"])}

    def renamed_definition():
        @G.component
        def prepare_v2(seed: int, n_rows: int, shift: float, train_v2: G.Output[G.Dataset], valid_v2: G.Output[G.Dataset]):
            pass

        @G.pipeline
        def broken(seed: int = 1, n_rows: int = 4000, shift: float = 1.0):
            p = prepare_v2(seed=seed, n_rows=n_rows, shift=shift)
            P.train(train_data=p.outputs["train_data"])

    try:
        renamed_definition()
        out["A_dag"] = {"raised": None}
    except KeyError as e:
        out["A_dag"] = {"raised": "KeyError", "message": str(e)[:160], "tasks_run": 0}

    # B: prepare가 실패. 어제 파일이 남아 있는 폴더에서
    out["B_glue"] = {}
    for strict in ("", "e", "pipefail"):
        w = tmp / f"B_{strict or 'none'}"
        GL.run_glue(w, **YESTERDAY)
        b = GL.run_glue(w, **TODAY, strict=strict, extra_env={"CH07_PREP_FAIL": "1"})
        out["B_glue"][strict or "none"] = {"exit_code": b["exit_code"], "metrics_after": brief(b["metrics"])}
    os.environ["CH07_PREP_FAIL"] = "1"
    try:
        run = G.run(ir, TODAY, tmp / "B_dag")
    finally:
        os.environ.pop("CH07_PREP_FAIL")
    out["B_dag"] = {"state": run["state"], "statuses": {t["task"]: t["status"] for t in run["tasks"]}, "metrics": _metrics_of(run)}

    # C: 두 실행(A=어제 설정, B=오늘 설정)이 한 폴더를 같이 쓰는 20가지 순서
    glue_bad = glue_mixed = dag_bad = 0
    orders = list(GL.interleavings())
    for i, order in enumerate(orders):
        w = tmp / f"C_glue_{i}"
        w.mkdir()
        steps = {"A": GL.shared_folder_steps(w, "A", **YESTERDAY), "B": GL.shared_folder_steps(w, "B", **TODAY)}
        for who, k in order:
            steps[who][k]()
        bad = False
        for who, cfg in (("A", YESTERDAY), ("B", TODAY)):
            m = json.loads((w / f"metrics_{who}.json").read_text())
            bad |= m["trained_on"]["seed"] != cfg["seed"] or m["evaluated_on"]["seed"] != cfg["seed"]
            glue_mixed += m["trained_on"]["seed"] != m["evaluated_on"]["seed"]
        glue_bad += bad
        root = tmp / f"C_dag_{i}"
        gens = {"A": G.run_steps(ir, YESTERDAY, root, run_id="runA"), "B": G.run_steps(ir, TODAY, root, run_id="runB")}
        logs = {"A": [], "B": []}
        for who, _ in order:
            logs[who].append(next(gens[who]))
        bad = False
        for who, cfg in (("A", YESTERDAY), ("B", TODAY)):
            ev = logs[who][-1]
            m = json.loads(Path(ev["outputs"]["metrics"]["uri"] + ".json").read_text())
            bad |= m["trained_on"]["seed"] != cfg["seed"] or m["evaluated_on"]["seed"] != cfg["seed"]
        dag_bad += bad
    out["C"] = {"orders": len(orders), "glue_contaminated_orders": glue_bad, "glue_runs_train_eval_mismatch": glue_mixed,
                "dag_contaminated_orders": dag_bad}
    return out


def e2(tmp):
    ir = G.compile(P.churn_pipeline)
    runs = [G.run(ir, TODAY, tmp / "E2") for _ in range(3)]
    ms = [_metrics_of(r) for r in runs]
    uris = {r["tasks"][0]["outputs"]["train_data"]["uri"] for r in runs}
    shas = {r["tasks"][0]["outputs"]["train_data"]["sha"] for r in runs}
    return {"accuracy": [r4(m["accuracy"]) for m in ms], "distinct_train_data_uris": len(uris), "distinct_train_data_sha": len(shas),
            "train_data_sha": sorted(shas)[0], "uri_pattern": "<pipeline_root>/<run_id>/prepare/train_data"}


def _timed(fn, repeats=5):
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def e3_e4(tmp):
    big = dict(TODAY, n_rows=200_000)
    ir = G.compile(P.churn_pipeline)
    res = {"n_rows": big["n_rows"], "repeats": 5}
    res["glue_bash_sec"] = r4(_timed(lambda: _glue_big(tmp)))
    for iso in ("inprocess", "subprocess"):
        per = {"prepare": [], "train": [], "evaluate": []}
        tot = []
        for _ in range(5):
            t0 = time.perf_counter()
            run = G.run(ir, big, tmp / f"E3_{iso}", isolation=iso)
            tot.append(time.perf_counter() - t0)
            for t in run["tasks"]:
                per[t["task"]].append(t["seconds"])
        res[iso] = {"total_sec": r4(statistics.median(tot)), "task_sec": {k: r4(statistics.median(v)) for k, v in per.items()}}
    # 손으로 따라 하는 작은 크기(4,000행)도 같이
    small = []
    for iso in ("inprocess", "subprocess"):
        tot = []
        for _ in range(5):
            t0 = time.perf_counter()
            G.run(ir, TODAY, tmp / f"E3s_{iso}", isolation=iso)
            tot.append(time.perf_counter() - t0)
        small.append(r4(statistics.median(tot)))
    res["small_4000_rows_total_sec"] = {"inprocess": small[0], "subprocess": small[1]}
    # E4: 평가 문턱만 0.5 -> 0.4. 캐시가 없으면 세 task를 모두 다시 돈다
    e4 = {}
    for iso in ("inprocess", "subprocess"):
        t = res[iso]["task_sec"]
        full = t["prepare"] + t["train"] + t["evaluate"]
        e4[iso] = {"full_rerun_sec": r4(full), "evaluate_only_sec": t["evaluate"], "wasted_share": r4((t["prepare"] + t["train"]) / full)}
    run04 = G.run(ir, dict(big, threshold=0.4), tmp / "E4")
    run05 = G.run(ir, big, tmp / "E4")
    m4, m5 = _metrics_of(run04), _metrics_of(run05)
    e4["threshold_0.5"] = brief(m5)
    e4["threshold_0.4"] = brief(m4)
    e4["same_train_data_sha"] = run04["tasks"][0]["outputs"]["train_data"]["sha"] == run05["tasks"][0]["outputs"]["train_data"]["sha"]
    e4["same_model_sha"] = run04["tasks"][1]["outputs"]["model"]["sha"] == run05["tasks"][1]["outputs"]["model"]["sha"]
    return res, e4


def _glue_big(tmp):
    import subprocess
    import sys
    w = tmp / "E3_glue"
    env = {**os.environ, "PY": sys.executable, "OMP_NUM_THREADS": "2"}
    # run_all.sh는 행 수를 받지 않으므로, 같은 세 명령을 행 20만 개로 직접 잇는다(스크립트와 같은 모양)
    S = str(HERE / "pipelines_ch07_steps.py")
    cmds = [
        [sys.executable, S, "prepare", "--seed", "1", "--shift", "1.0", "--n-rows", "200000", "--outdir", str(w)],
        [sys.executable, S, "train", "--data", str(w / "train.npz"), "--out", str(w / "model.json")],
        [sys.executable, S, "evaluate", "--model", str(w / "model.json"), "--data", str(w / "valid.npz"), "--out", str(w / "metrics.json")],
    ]
    for c in cmds:
        subprocess.run(c, env=env, check=True, capture_output=True)


def e5(tmp):
    ir = G.compile(P.churn_pipeline)
    rn = G.run(ir, dict(TODAY, impl="numpy"), tmp / "E5")
    rt = G.run(ir, dict(TODAY, impl="torch"), tmp / "E5")
    mn = json.loads(Path(rn["tasks"][1]["outputs"]["model"]["uri"]).read_text())
    mt = json.loads(Path(rt["tasks"][1]["outputs"]["model"]["uri"]).read_text())
    a, b = _metrics_of(rn), _metrics_of(rt)
    return {"accuracy_numpy": r4(a["accuracy"]), "accuracy_torch": r4(b["accuracy"]),
            "max_weight_diff": float(np.max(np.abs(np.array(mn["w"]) - np.array(mt["w"])))),
            "bias_diff": float(abs(mn["b"] - mt["b"])),
            "ir_identical": G.compile(P.churn_pipeline) == ir}


def e6():
    out = {"build_min": 8.0, "queue_min_mean": 5.0, "task_gap_min": 3.0, "n_commits": 2000}
    for gap in (120, 30, 10):
        out[f"gap_{gap}"] = {mode: {k: r4(v) for k, v in R.simulate(mode, gap).items()} for mode in ("latest", "sha")}
    return out


def e7():
    text = (HERE / "Jenkinsfile").read_text()
    broken = {
        "latest_tag": text.replace("${env.GIT_COMMIT}", "latest"),
        "train_in_jenkins": text.replace("sh 'python ci/compile_pipeline.py", "sh 'python -m churn.train'\n        sh 'python ci/compile_pipeline.py"),
        "gate_reads_local": text.replace("--metrics results/kfp_metrics.json", "--metrics results/local_metrics.json"),
    }
    return {"current": JF.lint(text), **{k: JF.lint(v) for k, v in broken.items()}}


def ir_compare():
    real_path = RES / "kfp_ir_2.17.0.json"
    if not real_path.exists():
        return None
    real = json.loads(real_path.read_text())
    mine = G.compile(P.churn_pipeline)
    rt, mt = real["root"]["dag"]["tasks"], mine["root"]["dag"]["tasks"]
    return {
        "tasks_real": sorted(rt), "tasks_mine": sorted(mt),
        "edges_real": G.edges(real), "edges_match": G.edges(real) == G.edges(mine),
        "dependent_tasks_match": all(rt[n].get("dependentTasks", []) == mt[n].get("dependentTasks", []) for n in rt),
        "components": len(real["components"]), "edges": len(G.edges(real)),
        "topo_order_real": G.toposort(real),
        "sdk_version": real.get("sdkVersion"), "schema_version": real.get("schemaVersion"),
    }


def main():
    t0 = time.time()
    tmp = Path(tempfile.mkdtemp(prefix="ch07-"))
    try:
        res = {
            "env": {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
                    "numpy": np.__version__, "torch": torch.__version__.split("+")[0]},
            "checked": "2026-10-08", "threads": 2, "today": TODAY, "yesterday": YESTERDAY,
            "setup": {"n_rows": 4000, "train_rows": 3000, "valid_rows": 1000, "timing_rows": 200_000, "glue_steps": 3},
        }
        res["E1"] = e1(tmp)
        res["E2"] = e2(tmp)
        res["E3"], res["E4"] = e3_e4(tmp)
        res["E5"] = e5(tmp)
        res["E6"] = e6()
        res["E7"] = e7()
        res["IR"] = ir_compare()
        kl = RES / "ch07_kfp_local.json"
        if kl.exists():
            k = json.loads(kl.read_text())
            res["KFP_local"] = {
                "kfp": k["kfp"], "seconds": [r["seconds"] for r in k["runs"]], "accuracy": [r["accuracy"] for r in k["runs"]],
                "distinct_run_dirs": len({r["run_dir"] for r in k["runs"]}), "files": k["runs"][0]["files"],
                "trained_on_sha": k["runs"][0]["trained_on"]["sha"], "wiring": k["wiring"],
                "prepare_failure": k["prepare_failure"], "client_signatures": k["client_signatures"],
            }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res["seconds"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: res[k] for k in ("E1", "E2", "E4", "E5", "E6", "E7")}, ensure_ascii=False, indent=1))
    print("E3", json.dumps(res["E3"]), "IR", json.dumps(res["IR"], ensure_ascii=False))
    print("seconds", res["seconds"])


if __name__ == "__main__":
    main()
