"""8장 비교 실험. `uv run python pipelines/ch08_lineage/experiments.py` 로 실행한다(네트워크·쿠버네티스·kfp 불필요).

E1 리니지 없이 모델 파일 하나만 남았을 때: 파일 안의 정보, 후보를 다시 학습해 맞혀 보기(생성기가 바뀌기 전/후),
   리니지 저장소로 한 번에 조회하기
E2 캐시 함정: 데이터 규칙(source_version)이 이미지 안에서 바뀌었는데 캐시 키가 못 보는 경우를 다섯 가지 설정으로
E3 캐시 켬/끔: 평가 문턱만 바꾼 run의 시간(행 20만 개, 3회 평균)
E4 '필요한 단계만 캐시 끄기'의 연쇄: prepare만 끄면 내용이 같아도 뒤 단계가 다시 도는가(ID 키 vs 내용 키)
E5 리니지 조회 시간: run 10, 50, 500개
E6 몸통 바꾸기(NumPy -> PyTorch): 정확도는 같아도 다른 모델 파일, 리니지에 impl이 남는지
E7 영향 범위: 문제 있는 학습 데이터 하나에서 아래로 따라가 영향받은 모델·지표·run 찾기

결과는 results/ch08.json 에 저장하고, 웹 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import json  # noqa: E402
import platform  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pipelines_ch08_data as D  # noqa: E402
import pipelines_ch08_model_np as M  # noqa: E402
import pipelines_ch08_runner as R  # noqa: E402
import pipelines_ch08_steps as S  # noqa: E402
import pipelines_ch08_store as ST  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch08.json"
IMG_V1 = {"source_version_default": "v1", "build": "commit-a1"}
IMG_V2 = {"source_version_default": "v2", "build": "commit-b2"}


def r4(x):
    return round(float(x), 4)


def statuses(run):
    return {t["task"]: t["status"] for t in run["tasks"]}


def truth_train_hash(seed, n_rows, shift, version):
    t = D.make_table(seed, n_rows, shift, version)
    n = len(t[D.LABEL])
    tr, _ = D.split(t, n * 3 // 4, seed)
    return D.dataset_hash(tr)


# ---------------- E1 ----------------
def e1(tmp):
    st, reg = ST.Store(), R.Registry()
    ref = reg.ref(digest=reg.push("commit-a1", IMG_V1))
    configs = [{"seed": 0, "shift": 0.0}, {"seed": 1, "shift": 1.0}, {"seed": 2, "shift": 0.5},
               {"seed": 3, "shift": 0.0}, {"seed": 4, "shift": 1.0}]
    runs = [R.run(st, reg, tmp, ref, args=c, caching=False, run_id=f"run-{i}") for i, c in enumerate(configs)]
    deployed = runs[2]
    model_path = st.artifact(deployed["artifacts"]["train.model"])["uri"]
    file_keys = sorted(json.loads(Path(model_path).read_text()))
    w_dep, b_dep, _ = S.load_model_weights(model_path)

    def brute_force(version):
        t0 = time.perf_counter()
        matches = []
        for i, c in enumerate(configs):
            t = D.make_table(c["seed"], 4000, c["shift"], version)
            n = len(t[D.LABEL])
            tr, _ = D.split(t, n * 3 // 4, c["seed"])
            m = M.train(tr)
            if np.max(np.abs(np.array(m["w"]) - w_dep)) < 1e-12 and abs(m["b"] - b_dep) < 1e-12:
                matches.append(i)
        return {"candidates": len(configs), "matches": matches, "seconds": r4(time.perf_counter() - t0)}

    before = brute_force("v1")
    after = brute_force("v2")  # 데이터 규칙이 바뀐 뒤 지금 코드로 다시 만들면
    t0 = time.perf_counter()
    tr = st.trace(deployed["artifacts"]["train.model"])
    trace_sec = time.perf_counter() - t0
    c = configs[2]
    independent = truth_train_hash(c["seed"], 4000, c["shift"], "v1")
    return {
        "configs": configs, "deployed_run": "run-2", "model_file_keys": file_keys,
        "brute_force_same_code": before, "brute_force_after_source_change": after,
        "trace": tr, "trace_ms": r4(trace_sec * 1000),
        "trace_hash_matches_independent": tr["train_data"]["dataset_hash"] == independent,
    }


# ---------------- E2 ----------------
def e2(tmp):
    out = {}
    intended_v2 = truth_train_hash(1, 4000, 1.0, "v2")
    v1_hash = truth_train_hash(1, 4000, 1.0, "v1")
    truth_v2_acc = None

    def scenario(name, policy, ref_before, ref_after, push_after, args_before=None, args_after=None,
                 env_before=None, env_after=None, intended="v2"):
        st, reg = ST.Store(), R.Registry()
        d1 = reg.push("latest", IMG_V1)
        reg.push("commit-a1", IMG_V1)
        r1 = R.run(st, reg, tmp, ref_before(reg, d1), args=args_before, policy=policy, env=env_before, run_id=f"{name}-1")
        d2 = push_after(reg)
        r2 = R.run(st, reg, tmp, ref_after(reg, d2), args=args_after, policy=policy, env=env_after, run_id=f"{name}-2")
        data = R.data_of(st, r2)
        met = R.metrics_of(st, r2)
        audit = audit_cached(st, r2)
        res = {"policy": policy, "run2_status": statuses(r2), "run2_image_ref": r2_ref(st, r2),
               "served_source_version": data["source_version"], "intended_source_version": intended,
               "served_hash": data["dataset_hash"][:12],
               "hash_is_intended": data["dataset_hash"] == (intended_v2 if intended == "v2" else v1_hash),
               "run2_accuracy": met["accuracy"], "run2_rows": data["rows"], "cached_across_image_or_env_change": audit,
               "fingerprint_same_as_run1": r1["tasks"][0]["fingerprint"] == r2["tasks"][0]["fingerprint"]}
        out[name] = res
        return st, r2

    latest = lambda reg, d: reg.ref("latest")  # noqa: E731
    by_digest = lambda reg, d: reg.ref(digest=d)  # noqa: E731
    push_latest_v2 = lambda reg: reg.push("latest", IMG_V2)  # noqa: E731

    st_a, r_a = scenario("A_latest_tag", "kfp-2.17.2", latest, latest, push_latest_v2)
    scenario("B_digest_ref", "kfp-2.17.2", by_digest, by_digest, push_latest_v2)
    scenario("C_param_v2", "kfp-2.17.2", latest, latest, push_latest_v2,
             args_before={"data_version": "v1"}, args_after={"data_version": "v2"})
    scenario("C_param_v1_kept", "kfp-2.17.2", latest, latest, push_latest_v2,
             args_before={"data_version": "v1"}, args_after={"data_version": "v1"}, intended="v1")
    same_img = lambda reg: reg.push("latest", IMG_V1)  # noqa: E731
    scenario("D_env_kfp_2.17.2", "kfp-2.17.2", latest, latest, same_img,
             env_before={"prepare": {"SOURCE_VERSION": "v1"}}, env_after={"prepare": {"SOURCE_VERSION": "v2"}})
    scenario("D_env_kfp_master", "kfp-master", latest, latest, same_img,
             env_before={"prepare": {"SOURCE_VERSION": "v1"}}, env_after={"prepare": {"SOURCE_VERSION": "v2"}})
    scenario("E_content_key", "content", latest, latest, push_latest_v2)

    # 진짜 오늘(v2) 데이터로 처음부터 돌렸다면
    st, reg = ST.Store(), R.Registry()
    d = reg.push("x", IMG_V2)
    rt = R.run(st, reg, tmp, reg.ref(digest=d), caching=False, run_id="truth-v2")
    truth_v2_acc = R.metrics_of(st, rt)
    out["truth_v2"] = {"accuracy": truth_v2_acc["accuracy"], "churn_rate": truth_v2_acc["churn_rate"],
                       "rows": R.data_of(st, rt)["rows"], "hash": intended_v2[:12]}
    out["truth_v1"] = {"hash": v1_hash[:12]}
    m1 = R.metrics_of(st_a, r_a)
    out["A_latest_tag"]["run2_churn_rate"] = m1["churn_rate"]
    return out


def r2_ref(st, r):
    return st.execution(r["tasks"][0]["execution_id"])["props"]["image_ref"]


def audit_cached(st, run_out):
    """리니지로 하는 사후 점검: CACHED 실행이 뜰 때 본 이미지 다이제스트·환경 변수가 결과를 처음 만든 실행과 다르면 표시. 표시됐다고 다 틀린 건 아니다(C_param_v1_kept는 표시되지만 맞는 재사용이다)."""
    found = []
    for t in run_out["tasks"]:
        e = st.execution(t["execution_id"])
        if e["state"] == "CACHED":
            src = st.execution(e["cached_from"])
            if src["props"]["image_digest"] != e["props"]["image_digest"] or src["props"]["env"] != e["props"]["env"]:
                found.append(t["task"])
    return found


# ---------------- E3 / E4 ----------------
def _timed_pair(tmp, n_rows, reps, warm_policy, second):
    rows = []
    for i in range(reps):
        st, reg = ST.Store(), R.Registry()
        ref = reg.ref(digest=reg.push("commit-a1", IMG_V1))
        R.run(st, reg, tmp, ref, args={"n_rows": n_rows}, policy=warm_policy, run_id=f"warm-{i}")
        rows.append(second(st, reg, ref, i))
    return rows


def e3(tmp, n_rows=200_000, reps=3):
    def off(st, reg, ref, i):
        return R.run(st, reg, tmp, ref, args={"n_rows": n_rows, "threshold": 0.4}, caching=False, run_id=f"off-{i}")

    def on(st, reg, ref, i):
        return R.run(st, reg, tmp, ref, args={"n_rows": n_rows, "threshold": 0.4}, caching=True, run_id=f"on-{i}")

    res = {}
    for label, fn in (("cache_off", off), ("cache_on", on)):
        runs = _timed_pair(tmp, n_rows, reps, "kfp-2.17.2", fn)
        res[label] = {
            "total_sec_each": [r4(r["seconds"]) for r in runs],
            "total_sec_mean": r4(statistics.mean(r["seconds"] for r in runs)),
            "task_sec_mean": {t: r4(statistics.mean(x["seconds"] for r in runs for x in r["tasks"] if x["task"] == t))
                              for t in ("prepare", "train", "evaluate")},
            "status": statuses(runs[-1]),
        }
    res["n_rows"], res["reps"] = n_rows, reps
    res["saved_share"] = r4(1 - res["cache_on"]["total_sec_mean"] / res["cache_off"]["total_sec_mean"])
    return res


def e4(tmp, n_rows=200_000, reps=3):
    res = {}
    for policy in ("kfp-2.17.2", "content"):
        def second(st, reg, ref, i, policy=policy):
            return R.run(st, reg, tmp, ref, args={"n_rows": n_rows}, policy=policy,
                         caching={"prepare": False}, run_id=f"cascade-{policy}-{i}")
        runs = _timed_pair(tmp, n_rows, reps, policy, second)
        res[policy] = {
            "status": statuses(runs[-1]),
            "total_sec_mean": r4(statistics.mean(r["seconds"] for r in runs)),
        }
    res["n_rows"], res["reps"] = n_rows, reps
    # 내용이 정말 같은지: 따로 한 번 더 돌려 prepare 출력 지문 비교
    st, reg = ST.Store(), R.Registry()
    ref = reg.ref(digest=reg.push("commit-a1", IMG_V1))
    a = R.run(st, reg, tmp, ref, run_id="same-1")
    b = R.run(st, reg, tmp, ref, caching={"prepare": False}, run_id="same-2")
    aa, bb = st.artifact(a["artifacts"]["prepare.train_data"]), st.artifact(b["artifacts"]["prepare.train_data"])
    res["prepare_output"] = {"ids": [aa["id"], bb["id"]], "same_content_sha": aa["sha"] == bb["sha"],
                             "same_dataset_hash": aa["props"]["dataset_hash"] == bb["props"]["dataset_hash"]}
    return res


# ---------------- E5 ----------------
def e5(tmp, sizes=(10, 50, 500), repeats=200):
    res = {}
    for n in sizes:
        st, reg = ST.Store(), R.Registry()
        ref = reg.ref(digest=reg.push("commit-a1", IMG_V1))
        last = None
        for i in range(n):
            last = R.run(st, reg, tmp, ref, args={"seed": i, "n_rows": 400, "shift": (i % 5) * 0.25}, caching=False,
                         run_id=f"r{i:04d}")
        mid = last["artifacts"]["train.model"]
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            st.trace(mid)
            ts.append(time.perf_counter() - t0)
        a_sql, e_sql = st.ancestors_sql(mid)
        a_bfs, e_bfs = st.ancestors_bfs(mid)
        res[str(n)] = {"store": st.counts(), "trace_ms_median": r4(statistics.median(ts) * 1000),
                       "ancestors": {"artifacts": len(a_sql), "executions": len(e_sql)},
                       "sql_equals_bfs": (a_sql, e_sql) == (a_bfs, e_bfs)}
    res["repeats"] = repeats
    return res


# ---------------- E6 ----------------
def e6(tmp):
    st, reg = ST.Store(), R.Registry()
    ref = reg.ref(digest=reg.push("commit-a1", IMG_V1))
    a = R.run(st, reg, tmp, ref, args={"impl": "numpy"}, run_id="np")
    b = R.run(st, reg, tmp, ref, args={"impl": "torch"}, run_id="torch")
    ma, mb = (st.artifact(r["artifacts"]["train.model"]) for r in (a, b))
    wa, ba, _ = S.load_model_weights(ma["uri"])
    wb, bb, _ = S.load_model_weights(mb["uri"])
    return {
        "status_torch_run": statuses(b),
        "same_train_data_artifact": a["artifacts"]["prepare.train_data"] == b["artifacts"]["prepare.train_data"],
        "accuracy": [R.metrics_of(st, a)["accuracy"], R.metrics_of(st, b)["accuracy"]],
        "model_sha": [ma["sha"][:12], mb["sha"][:12]],
        "max_weight_diff": float(np.max(np.abs(wa - wb))), "bias_diff": abs(ba - bb),
        "trace_impl": [st.trace(ma["id"])["made_by"]["params"]["impl"], st.trace(mb["id"])["made_by"]["params"]["impl"]],
    }


# ---------------- E7 ----------------
def e7(tmp):
    """같은 학습 데이터를 쓰는 run 4개(둘은 캐시로 재사용, 하나는 문턱만 다름) + 다른 데이터 run 1개."""
    st, reg = ST.Store(), R.Registry()
    ref = reg.ref(digest=reg.push("commit-a1", IMG_V1))
    runs = [R.run(st, reg, tmp, ref, run_id="day1"),
            R.run(st, reg, tmp, ref, run_id="day2"),
            R.run(st, reg, tmp, ref, args={"threshold": 0.4}, run_id="day3"),
            R.run(st, reg, tmp, ref, args={"impl": "torch"}, run_id="day4"),
            R.run(st, reg, tmp, ref, args={"seed": 7}, run_id="day5")]
    bad = runs[0]["artifacts"]["prepare.train_data"]
    arts, exes = st.descendants_sql(bad)
    run_ids = sorted({st.execution(e)["run_id"] for e in exes})
    types = {}
    for a in arts - {bad}:
        t = st.artifact(a)["type"]
        types[t] = types.get(t, 0) + 1
    return {"runs": [{"run_id": r["run_id"], "status": statuses(r)} for r in runs],
            "affected_runs": run_ids, "affected_artifacts_by_type": types,
            "affected_executions": len(exes), "unaffected_runs": sorted({r["run_id"] for r in runs} - set(run_ids))}


def main():
    t_all = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix="ch08-"))
    try:
        res = {
            "env": {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
                    "platform": f"{platform.system()} {platform.machine()}", "threads": 2},
            "E1": e1(tmp / "e1"), "E2": e2(tmp / "e2"), "E3": e3(tmp / "e3"), "E4": e4(tmp / "e4"),
            "E5": e5(tmp / "e5"), "E6": e6(tmp / "e6"), "E7": e7(tmp / "e7"),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    res["total_seconds"] = round(time.perf_counter() - t_all, 1)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
