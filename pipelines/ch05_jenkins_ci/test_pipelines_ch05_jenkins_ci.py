"""5장 검증. `uv run pytest -q pipelines/ch05_jenkins_ci` 로 실행한다. git이 필요하고, 네트워크·Jenkins·Docker 없이 돈다."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import pipelines_ch05_history as hist
import pipelines_ch05_jenkins as jk
import pipelines_ch05_model_torch as mtt
import pipelines_ch05_project as pr
import pipelines_ch05_scenarios as ex

HERE = Path(__file__).parent
JF = (HERE / "Jenkinsfile").read_text()


# ---------- Jenkinsfile 읽기와 규칙 ----------

def test_parse_jenkinsfile():
    p = jk.parse(JF)
    assert [s["name"] for s in p["stages"]] == ["Checkout", "Lint", "Unit/Data tests", "Build image", "Model perf test", "Push image"]
    assert p["options"] == ["skipStagesAfterUnstable"]
    assert p["environment"] == {"OMP_NUM_THREADS": "2"}
    assert [n for n, a, b in p["post"]["always"]] == ["junit", "archiveArtifacts"]


def test_our_jenkinsfile_passes_lint():
    assert jk.lint(JF) == []


@pytest.mark.parametrize("name, expect", [
    ("B_true_junit_post_only_guard", "junit으로 읽지 않는다"),
    ("D_true_junit_in_stage_no_guard", "UNSTABLE 빌드에서도"),
    ("E_true_no_junit", "post { always"),
])
def test_lint_catches_traps(name, expect):
    assert any(expect in p for p in jk.lint(ex.variants()[name]))


@pytest.mark.parametrize("result, expected", [(None, True), ("SUCCESS", True), ("UNSTABLE", False), ("FAILURE", False)])
def test_when_guard(result, expected):
    raw = jk.parse(JF)["stages"][-1]["when"][0][2]
    assert jk.eval_when(raw, result) is expected


def test_unknown_step_is_not_silently_ignored(tmp_path):
    text = "pipeline {\n agent any\n stages {\n  stage('x') { steps { bat 'dir' } }\n }\n}\n"
    with pytest.raises(NotImplementedError):
        jk.run_pipeline(text, tmp_path)


# ---------- 실패하는 최소 예제: 진짜 git, 진짜 pytest ----------

@pytest.fixture(scope="module")
def two_branch(tmp_path_factory):
    d = tmp_path_factory.mktemp("e1")
    repo = d / "repo"
    pr.make_two_branches(repo)
    out = {b: ex.run_at(repo, b, d, b.replace("-", "_")) for b in ("branch-a", "branch-b")}
    changed = {b: set(pr.git(repo, "diff", "--name-only", "main", b).stdout.split()) for b in ("branch-a", "branch-b")}
    _, conflicts = pr.merge_both(repo)
    out["merged"] = ex.run_at(repo, "main", d, "merged")
    out["changed"], out["conflicts"] = changed, conflicts
    return out


def test_each_branch_alone_is_green(two_branch):
    for b in ("branch-a", "branch-b"):
        r = two_branch[b]
        assert r["result"] == "SUCCESS" and all(t["ok"] for t in r["tests"]) and r["pushed"]


def test_merge_is_textually_clean(two_branch):
    assert two_branch["conflicts"] == []
    assert two_branch["changed"]["branch-a"].isdisjoint(two_branch["changed"]["branch-b"])


def test_merged_commit_fails_in_unit_stage(two_branch):
    r = two_branch["merged"]
    status = {s["name"]: s["status"] for s in r["stages"]}
    assert r["result"] == "FAILURE" and status["Unit/Data tests"] == "FAILURE"
    assert status["Push image"].startswith("SKIPPED") and r["pushed"] == []
    failed = [t for t in r["tests"] if not t["ok"]]
    assert [t["name"] for t in failed] == ["tests.test_extra::test_fee_per_call_positive"]
    assert "KeyError" in failed[0]["message"] and "monthly_fee" in failed[0]["message"]
    assert "always" in r["post_ran"]


def test_swallowed_failure_pushes_unless_guarded(tmp_path):
    repo = tmp_path / "repo"
    pr.make_two_branches(repo)
    pr.merge_both(repo)
    v = ex.variants()
    b = ex.run_at(repo, "main", tmp_path, "b", jenkinsfile=v["B_true_junit_post_only_guard"])
    c = ex.run_at(repo, "main", tmp_path, "c", jenkinsfile=v["C_true_junit_in_stage_guard"])
    assert b["result"] == "UNSTABLE" and b["pushed"]          # 결과를 늦게 읽으면 막는 장치가 있어도 push된다
    assert c["result"] == "UNSTABLE" and c["pushed"] == []    # 같은 단계에서 읽으면 when이 막는다


# ---------- 합치는 간격 ----------

def test_resolve_registry_keeps_both(tmp_path):
    p = tmp_path / "registry.txt"
    p.write_text("a\n<<<<<<< HEAD\nb\n=======\nc\n>>>>>>> main\n")
    assert hist.resolve(p) == (1, 2)
    assert p.read_text() == "a\nb\nc\n"


def test_gate_flags_missing_column(tmp_path):
    main, wts = hist.setup(tmp_path / "r")
    (main / "features" / "f_x.txt").write_text("uses: income\n")
    with open(main / "registry.txt", "a") as fh:
        fh.write("f_x\n")
    assert hist.gate(main) == ["f_x"]


def test_late_integration_finds_more_breaks_later(tmp_path):
    daily = hist.simulate(1, seed=0, root=tmp_path / "d")
    late = hist.simulate(10, seed=0, branch_ci=True, root=tmp_path / "l")
    assert late["broken_found"] > daily["broken_found"]
    assert late["max_delay_days"] > daily["max_delay_days"] == 0
    assert late["branch_ci_runs"] == 80 and late["branch_ci_red"] == 0   # 브랜치 CI는 의미 충돌을 못 본다
    assert daily["final_main_broken"] == late["final_main_broken"] == 0


def test_history_is_deterministic(tmp_path):
    a = hist.simulate(5, seed=1, root=tmp_path / "a")
    b = hist.simulate(5, seed=1, root=tmp_path / "b")
    assert a == b


# ---------- 성능 하한과 재현 ----------

def test_numpy_torch_same_gate(tmp_path):
    m = pr.load_modules(tmp_path / "p")
    t = m["data"].make_table(0, 1200)
    X, y = m["features"].build_features(t), t[m["data"].LABEL]
    w1, b1 = m["model"].fit_logistic(X, y)
    w2, b2 = mtt.fit_logistic(X, y)
    assert np.max(np.abs(w1 - w2)) < 1e-12 and abs(b1 - b2) < 1e-12


def test_seeded_gate_is_stable(tmp_path):
    m = pr.load_modules(tmp_path / "p")
    assert len({m["train"].run()["val_acc"] for _ in range(5)}) == 1


def test_results_file_consistent():
    res = json.loads((HERE / "results" / "ch05.json").read_text())
    assert res["E1"]["merged"]["result"] == "FAILURE" and res["E1"]["merge_textual_conflicts"] == []
    assert res["E3"]["B_true_junit_post_only_guard"]["pushed"] and not res["E3"]["C_true_junit_in_stage_guard"]["pushed"]
    s = res["E2"]["summary"]
    assert s["10"]["broken_total"] > s["1"]["broken_total"]
    assert res["E4"]["image_digests_equal"] and res["E4"]["metrics_identical"]
    assert res["E5"]["seeded"]["distinct"] == 1


def test_git_available():
    assert subprocess.run(["git", "--version"], capture_output=True).returncode == 0
