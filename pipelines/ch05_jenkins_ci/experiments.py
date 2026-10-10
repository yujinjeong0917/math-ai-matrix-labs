"""5장 비교 실험. `uv run python pipelines/ch05_jenkins_ci/experiments.py` 로 실행한다(git 필요, 네트워크·Jenkins·Docker 불필요).

E1 두 브랜치, 한 번의 합치기: A는 열 이름 변경(monthly_fee -> monthly_fee_krw), B는 그 열을 쓰는 새 특성.
   각 브랜치와 합친 커밋에서 Jenkinsfile을 흉내 러너로 실제로 돌려 단계별 결과와 실패 테스트 이름을 남긴다.
E2 합치는 간격(1, 2, 5, 10일)만 바꾼 2주 이력: 4명 x 10일 x 하루 2커밋, 시드 3개. 텍스트 충돌, 깨진 특성, 발견까지 걸린 날.
   10일 간격에는 브랜치 CI(브랜치 커밋마다 검사)를 켜서 검사가 몇 번 돌고 몇 번 빨간불이었는지도 센다.
E3 Jenkinsfile의 함정: 테스트 실패를 `|| true`로 삼킨 네 가지 변형을 같은 깨진 커밋에서 돌려 push가 일어나는지 본다.
E4 단계별 소요 시간과 재실행 일치: 정상 커밋에서 파이프라인을 3번 돌려 단계별 시간 평균, 지표·이미지 지문이 같은지.
E5 흔들리는 테스트: 성능 하한 0.80 검사를 시드 고정으로 50번, 실행마다 시드가 바뀌는 경우(시드 0~49로 대신)로 50번 돌린다.
   시드를 아예 안 주는 경우는 실행할 때마다 숫자가 달라 결과 파일에 남기지 않는다.
E6 NumPy와 PyTorch 대조: 같은 특성 행렬에서 두 구현의 가중치 차이와 성능 하한 판정.

결과는 results/ch05.json 에 저장하고, 웹 챕터 본문은 이 파일의 값을 그대로 인용한다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import statistics  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pipelines_ch05_history as hist  # noqa: E402
import pipelines_ch05_model_torch as mtt  # noqa: E402
import pipelines_ch05_project as pr  # noqa: E402
from pipelines_ch05_scenarios import run_at, short, variants  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent
OUT = HERE / "results" / "ch05.json"
INTERVALS = (1, 2, 5, 10)
SEEDS = (0, 1, 2)
FLOOR_FLAKY = 0.80
N_FLAKY = 50


def env_snapshot():
    return {"python": platform.python_version(), "machine": platform.machine(), "system": platform.system(),
            "numpy": np.__version__, "torch": torch.__version__,
            "git": subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()}


def e1_two_branches():
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "repo"
        shas = pr.make_two_branches(repo)
        out = {"shas": {k: v[:12] for k, v in shas.items()}}
        for br in ("main", "branch-a", "branch-b"):
            out[br] = short(run_at(repo, br, d, br))
        changed = {}
        for br in ("branch-a", "branch-b"):
            changed[br] = pr.git(repo, "diff", "--name-only", "main", br).stdout.split()
        out["changed_files"] = changed
        out["overlap_files"] = sorted(set(changed["branch-a"]) & set(changed["branch-b"]))
        merged, conflicts = pr.merge_both(repo)
        out["merge_textual_conflicts"] = conflicts
        out["merged"] = short(run_at(repo, "main", d, "merged"))
        # CI가 없다면: 합친 뒤 아무도 테스트를 돌리지 않고 다음 주 출시 직전에 처음 돌린다고 하자.
        # 그 사이 main에 커밋이 쌓이는 것은 E2에서 센다.
    return out


def e2_history():
    rows = []
    for T in INTERVALS:
        for s in SEEDS:
            with tempfile.TemporaryDirectory() as d:
                r = hist.simulate(T, s, branch_ci=(T == 10), root=Path(d) / "r")
            r.pop("delays_list", None)
            rows.append(r)
    summary = {}
    for T in INTERVALS:
        rs = [r for r in rows if r["interval"] == T]
        summary[str(T)] = {
            "integrations": rs[0]["integrations"],
            "merges_with_conflict": [r["merges_with_conflict"] for r in rs],
            "conflict_hunks": [r["conflict_hunks"] for r in rs],
            "conflict_lines": [r["conflict_lines"] for r in rs],
            "lines_per_hunk": [round(r["conflict_lines"] / r["conflict_hunks"], 2) if r["conflict_hunks"] else 0 for r in rs],
            "broken_found": [r["broken_found"] for r in rs],
            "broken_total": sum(r["broken_found"] for r in rs),
            "max_delay_days": [r["max_delay_days"] for r in rs],
            "delays_all": sorted(x for r in rs for x in r["delays"]),
            "mean_unsynced_at_merge": [round(statistics.mean(r["unsynced_changes_at_merge"]), 1) for r in rs],
            "branch_ci_runs": [r["branch_ci_runs"] for r in rs],
            "branch_ci_red": [r["branch_ci_red"] for r in rs],
            "main_red_after_round": [r["main_red_after_round"] for r in rs],
            "final_main_broken": [r["final_main_broken"] for r in rs],
        }
        d = summary[str(T)]["delays_all"]
        summary[str(T)]["mean_delay_days"] = round(statistics.mean(d), 2) if d else 0.0
    return {"setup": {"devs": hist.N_DEVS, "days": hist.DAYS, "commits_per_day": hist.COMMITS_PER_DAY,
                      "rename_days": list(hist.RENAME_DAYS), "seeds": list(SEEDS), "commits_total": rows[0]["commits"]},
            "rows": rows, "summary": summary}


def e3_traps():
    out = {}
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "repo"
        pr.make_two_branches(repo)
        pr.merge_both(repo)
        for name, text in variants().items():
            r = run_at(repo, "main", d, name, jenkinsfile=text)
            out[name] = short(r) | {"n_failed": sum(not t["ok"] for t in r["tests"])}
    return out


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def e4_timing(reps=3):
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d) / "repo"
        pr.make_repo(repo)
        runs = []
        for i in range(reps):
            r = run_at(repo, "main", d, f"run{i}")
            arch = Path(d) / f"archive-run{i}"
            runs.append({"result": r["result"], "stages": {s["name"]: s["seconds"] for s in r["stages"]},
                         "metrics": json.loads((arch / "metrics.json").read_text()) if (arch / "metrics.json").exists() else None,
                         "metrics_sha": sha_file(arch / "metrics.json") if (arch / "metrics.json").exists() else None,
                         "image": json.loads((arch / "image.json").read_text())["digest"],
                         "tests": [(t["name"], t["ok"]) for t in r["tests"]]})
    names = list(runs[0]["stages"])
    mean_s = {n: round(statistics.mean(r["stages"][n] for r in runs), 2) for n in names}
    return {"reps": reps, "stage_mean_seconds": mean_s, "total_mean_seconds": round(sum(statistics.mean(r["stages"][n] for r in runs) for n in names), 2),
            "results": [r["result"] for r in runs], "image_digests_equal": len({r["image"] for r in runs}) == 1,
            "image_digest": runs[0]["image"], "metrics_identical": len({r["metrics_sha"] for r in runs}) == 1,
            "metrics": runs[0]["metrics"], "tests_identical": len({json.dumps(r["tests"]) for r in runs}) == 1}


def e5_flaky():
    with tempfile.TemporaryDirectory() as d:
        m = pr.load_modules(Path(d) / "p")
        seeded = [m["train"].run()["val_acc"] for _ in range(N_FLAKY)]
        seeds_varied = [m["train"].run(data_seed=s, split_seed=s + 1)["val_acc"] for s in range(N_FLAKY)]

    def summ(xs):
        passes = [x >= FLOOR_FLAKY for x in xs]
        flips = sum(passes[i] != passes[i - 1] for i in range(1, len(passes)))
        return {"n": len(xs), "pass": sum(passes), "fail": len(xs) - sum(passes), "flips": flips,
                "min": min(xs), "max": max(xs), "mean": round(statistics.mean(xs), 4), "std": round(statistics.pstdev(xs), 4),
                "distinct": len(set(xs))}
    return {"floor": FLOOR_FLAKY, "seeded": summ(seeded), "seed_sweep_0_49": summ(seeds_varied)}


def e6_np_vs_torch():
    with tempfile.TemporaryDirectory() as d:
        m = pr.load_modules(Path(d) / "p")
        table = m["data"].make_table(0, 1200)
        X = m["features"].build_features(table)
        y = table[m["data"].LABEL]
        order = np.random.default_rng(1).permutation(1200)
        tr, va = order[:800], order[800:]
        w_np, b_np = m["model"].fit_logistic(X[tr], y[tr])
        w_t, b_t = mtt.fit_logistic(X[tr], y[tr])
        acc = lambda w, b: float(np.mean((m["model"].predict_proba(X[va], w, b) >= 0.5) == y[va]))  # noqa: E731
        return {"max_abs_w_diff": float(np.max(np.abs(w_np - w_t))), "b_diff": abs(b_np - b_t),
                "acc_np": acc(w_np, b_np), "acc_torch": acc(w_t, b_t), "floor": 0.75,
                "both_pass": acc(w_np, b_np) >= 0.75 and acc(w_t, b_t) >= 0.75, "train_run_acc": m["train"].run()["val_acc"]}


def main():
    res = {"env": env_snapshot(), "checked": "2026-10-08", "threads": 2}
    res["E1"] = e1_two_branches()
    print("E1", res["E1"]["merged"]["result"], res["E1"]["merge_textual_conflicts"])
    res["E2"] = e2_history()
    print("E2", {k: (v["broken_total"], v["conflict_hunks"]) for k, v in res["E2"]["summary"].items()})
    res["E3"] = e3_traps()
    print("E3", {k: (v["result"], v["pushed"]) for k, v in res["E3"].items()})
    res["E4"] = e4_timing()
    print("E4", res["E4"]["stage_mean_seconds"])
    res["E5"] = e5_flaky()
    print("E5", res["E5"])
    res["E6"] = e6_np_vs_torch()
    print("E6", res["E6"])
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")
    print("saved", OUT)


if __name__ == "__main__":
    main()
