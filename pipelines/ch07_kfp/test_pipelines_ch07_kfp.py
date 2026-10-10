"""7장 검증. `uv run pytest -q pipelines/ch07_kfp` 로 실행한다. 네트워크·Jenkins·쿠버네티스·kfp 없이 돈다.

실제 kfp 2.17.0과의 비교는 kfp_local_check.py가 만들어 둔 results/kfp_ir_2.17.0.json,
results/ch07_kfp_local.json을 읽어서 한다(이 두 파일은 저장소에 함께 들어 있다).
"""

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

import pipelines_ch07_dag as G
import pipelines_ch07_glue as GL
import pipelines_ch07_jenkinsfile as JF
import pipelines_ch07_pipeline as P
import pipelines_ch07_registry as R
import pipelines_ch07_steps as S

HERE = Path(__file__).parent
REAL_IR = json.loads((HERE / "results" / "kfp_ir_2.17.0.json").read_text())
KFP_LOCAL = json.loads((HERE / "results" / "ch07_kfp_local.json").read_text())
TODAY, YESTERDAY = {"seed": 1, "shift": 1.0}, {"seed": 0, "shift": 0.0}


@pytest.fixture(scope="module")
def ir():
    return G.compile(P.churn_pipeline)


def test_graph_matches_real_kfp_compile(ir):
    real, mine = REAL_IR["root"]["dag"]["tasks"], ir["root"]["dag"]["tasks"]
    assert sorted(real) == sorted(mine) == ["evaluate", "prepare", "train"]
    assert G.edges(REAL_IR) == G.edges(ir)
    assert len(G.edges(ir)) == 3
    for n in real:
        assert real[n].get("dependentTasks", []) == mine[n].get("dependentTasks", [])
        assert real[n]["inputs"].get("parameters", {}) == mine[n]["inputs"].get("parameters", {})
    for c, spec in REAL_IR["components"].items():
        for side in ("inputDefinitions", "outputDefinitions"):
            ra = {k: v["artifactType"]["schemaTitle"] for k, v in spec.get(side, {}).get("artifacts", {}).items()}
            ma = {k: v["artifactType"]["schemaTitle"] for k, v in ir["components"][c].get(side, {}).get("artifacts", {}).items()}
            assert ra == ma
        rp = {k: v["parameterType"] for k, v in spec.get("inputDefinitions", {}).get("parameters", {}).items()}
        mp = {k: v["parameterType"] for k, v in ir["components"][c].get("inputDefinitions", {}).get("parameters", {}).items()}
        assert rp == mp
    assert REAL_IR["sdkVersion"] == "kfp-2.17.0"


def test_kfp_file_and_mini_file_define_the_same_functions():
    def sigs(path):
        tree = ast.parse(Path(path).read_text())
        return {f.name: ast.dump(f) for f in tree.body if isinstance(f, ast.FunctionDef) and not f.name.startswith("_")}
    assert sigs(HERE / "kfp_pipeline.py") == sigs(HERE / "pipelines_ch07_pipeline.py")


def test_toposort_and_cycle(ir):
    assert G.toposort(ir) == ["prepare", "train", "evaluate"] == G.toposort(REAL_IR)
    bad = json.loads(json.dumps(ir))
    bad["root"]["dag"]["tasks"]["prepare"]["dependentTasks"] = ["evaluate"]
    with pytest.raises(ValueError):
        G.toposort(bad)


def test_wiring_mistakes_fail_before_anything_runs():
    @G.component
    def prep2(seed: int, out_v2: G.Output[G.Dataset]):
        pass

    @G.component
    def needs_int(x: int):
        pass

    with pytest.raises(KeyError):
        @G.pipeline
        def renamed(seed: int = 1):
            P.train(train_data=prep2(seed=seed).outputs["train_data"])
    with pytest.raises(TypeError):
        @G.pipeline
        def wrong_type(seed: int = 1):
            needs_int(x=prep2(seed=seed).outputs["out_v2"])
    with pytest.raises(TypeError):
        @G.pipeline
        def model_into_dataset(seed: int = 1):
            p = P.prepare(seed=seed, n_rows=100, shift=0.0)
            t = P.train(train_data=p.outputs["train_data"])
            P.evaluate(model=t.outputs["model"], valid_data=t.outputs["model"])
    # 실제 kfp 2.17.0도 같은 두 실수를 정의하는 순간 거절했다(kfp_local_check.py 기록)
    assert KFP_LOCAL["wiring"]["renamed_output"]["raised"] == "KeyError"
    assert KFP_LOCAL["wiring"]["type_mismatch"]["raised"] == "InconsistentTypeException"


def test_run_gives_same_numbers_as_direct_calls_and_new_uris(ir, tmp_path):
    S.prepare(1, 4000, 1.0, tmp_path / "tr", tmp_path / "va")
    S.train(tmp_path / "tr", tmp_path / "m")
    direct = S.evaluate(tmp_path / "m", tmp_path / "va", tmp_path / "x.json")
    r1, r2 = G.run(ir, TODAY, tmp_path / "root"), G.run(ir, TODAY, tmp_path / "root")
    m1 = json.loads(Path(r1["tasks"][2]["outputs"]["metrics"]["uri"] + ".json").read_text())
    assert m1["accuracy"] == direct["accuracy"] == KFP_LOCAL["runs"][0]["accuracy"]
    assert m1["trained_on"]["sha"] == KFP_LOCAL["runs"][0]["trained_on"]["sha"]
    u1, u2 = r1["tasks"][0]["outputs"]["train_data"]["uri"], r2["tasks"][0]["outputs"]["train_data"]["uri"]
    assert u1 != u2 and r1["tasks"][0]["outputs"]["train_data"]["sha"] == r2["tasks"][0]["outputs"]["train_data"]["sha"]
    assert r1["tasks"][1]["inputs"]["train_data"]["uri"] == u1  # 기록: train이 실제로 읽은 위치


def test_failed_task_skips_downstream(ir, tmp_path, monkeypatch):
    monkeypatch.setenv("CH07_PREP_FAIL", "1")
    r = G.run(ir, TODAY, tmp_path)
    assert r["state"] == "FAILED"
    assert [t["status"] for t in r["tasks"]] == ["FAILED", "SKIPPED", "SKIPPED"]
    assert KFP_LOCAL["prepare_failure"]["task_dirs"] == ["prepare"]


def test_save_keeps_the_exact_path(tmp_path):
    S.prepare(0, 100, 0.0, tmp_path / "train_data", tmp_path / "valid_data")
    assert (tmp_path / "train_data").is_file() and not (tmp_path / "train_data.npz").exists()


def test_glue_rename_is_silent(tmp_path):
    GL.run_glue(tmp_path, **YESTERDAY)
    r = GL.run_glue(tmp_path, **TODAY, extra_env={"CH07_PREP_SUFFIX": "_v2"})
    assert r["exit_code"] == 0
    assert r["metrics"]["trained_on"]["seed"] == 0 and r["metrics"]["evaluated_on"]["seed"] == 0


@pytest.mark.parametrize("strict,code", [("", 0), ("e", 0), ("pipefail", 1)])
def test_set_e_with_tee_hides_the_failure(tmp_path, strict, code):
    GL.run_glue(tmp_path, **YESTERDAY)
    r = GL.run_glue(tmp_path, **TODAY, strict=strict, extra_env={"CH07_PREP_FAIL": "1"})
    assert r["exit_code"] == code


def test_shared_folder_vs_unique_uris(ir, tmp_path):
    orders = list(GL.interleavings())
    assert len(orders) == 20
    glue_bad = dag_bad = 0
    for i, order in enumerate(orders):
        w = tmp_path / f"g{i}"
        w.mkdir()
        steps = {"A": GL.shared_folder_steps(w, "A", **YESTERDAY), "B": GL.shared_folder_steps(w, "B", **TODAY)}
        for who, k in order:
            steps[who][k]()
        glue_bad += any(json.loads((w / f"metrics_{who}.json").read_text())["evaluated_on"]["seed"] != cfg["seed"]
                        or json.loads((w / f"metrics_{who}.json").read_text())["trained_on"]["seed"] != cfg["seed"]
                        for who, cfg in (("A", YESTERDAY), ("B", TODAY)))
        gens = {"A": G.run_steps(ir, YESTERDAY, tmp_path / f"d{i}", run_id="a"), "B": G.run_steps(ir, TODAY, tmp_path / f"d{i}", run_id="b")}
        last = {}
        for who, _ in order:
            last[who] = next(gens[who])
        for who, cfg in (("A", YESTERDAY), ("B", TODAY)):
            m = json.loads(Path(last[who]["outputs"]["metrics"]["uri"] + ".json").read_text())
            dag_bad += m["trained_on"]["seed"] != cfg["seed"]
    assert (glue_bad, dag_bad) == (18, 0)


def test_component_body_swap_numpy_torch(ir, tmp_path):
    rn = G.run(ir, dict(TODAY, impl="numpy"), tmp_path)
    rt = G.run(ir, dict(TODAY, impl="torch"), tmp_path)
    wn = json.loads(Path(rn["tasks"][1]["outputs"]["model"]["uri"]).read_text())["w"]
    wt = json.loads(Path(rt["tasks"][1]["outputs"]["model"]["uri"]).read_text())["w"]
    assert np.allclose(wn, wt, atol=1e-12)
    a = [json.loads(Path(r["tasks"][2]["outputs"]["metrics"]["uri"] + ".json").read_text())["accuracy"] for r in (rn, rt)]
    assert a[0] == a[1]


def test_jenkinsfile_boundary_rules():
    text = (HERE / "Jenkinsfile").read_text()
    assert JF.lint(text) == []
    assert any("움직이는 태그" in p for p in JF.lint(text.replace("${env.GIT_COMMIT}", "latest")))
    assert any("학습" in p for p in JF.lint(text.replace("steps { sh 'python -m pytest", "steps { sh 'python -m churn.train'; sh 'python -m pytest")))
    assert any("먼저" in p for p in JF.lint(text.replace("stage('Gate')", "stage('Gate-early')").replace(
        "stage('Unit/Data tests')", "stage('Gate') { steps { sh 'python ci/decide.py --metrics results/kfp_metrics.json' } }\n    stage('Unit/Data tests')", 1)))


def test_submit_uses_real_kfp_argument_names():
    tree = ast.parse((HERE / "pipelines_ch07_submit.py").read_text())
    sig = KFP_LOCAL["client_signatures"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in sig:
            assert {k.arg for k in node.keywords} <= set(sig[node.func.attr])


def test_mutable_tag_simulation():
    for gap in (120, 30, 10):
        assert R.simulate("sha", gap)["wrong_image_run_rate"] == 0.0
    rates = [R.simulate("latest", g)["wrong_image_run_rate"] for g in (120, 30, 10)]
    assert 0 < rates[0] < rates[1] < rates[2]
    # 손으로 하는 어림: 다음 이미지가 '대기 + 마지막 task까지 6분' 안에 올라올 확률 1 - e^{-6/g} * g/(g+5)
    g = 120
    assert abs(rates[0] - (1 - np.exp(-6 / g) * g / (g + 5))) < 0.02
