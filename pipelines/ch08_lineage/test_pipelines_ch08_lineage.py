"""8장 검증. `uv run pytest -q pipelines/ch08_lineage` 로 실행한다. 네트워크·쿠버네티스·kfp 없이 돈다."""

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import pipelines_ch08_data as D  # noqa: E402
import pipelines_ch08_runner as R  # noqa: E402
import pipelines_ch08_steps as S  # noqa: E402
import pipelines_ch08_store as ST  # noqa: E402

torch.set_num_threads(2)
V1 = {"source_version_default": "v1", "build": "commit-a1"}
V2 = {"source_version_default": "v2", "build": "commit-b2"}


def statuses(run):
    return [t["status"] for t in run["tasks"]]


def train_hash(seed, n_rows, shift, version):
    t = D.make_table(seed, n_rows, shift, version)
    n = len(t[D.LABEL])
    tr, _ = D.split(t, n * 3 // 4, seed)
    return D.dataset_hash(tr)


@pytest.fixture
def env(tmp_path):
    st, reg = ST.Store(), R.Registry()
    reg.push("latest", V1)
    return st, reg, tmp_path


def test_source_versions_differ():
    assert train_hash(1, 4000, 1.0, "v1") != train_hash(1, 4000, 1.0, "v2")
    assert len(D.make_table(1, 4000, 1.0, "v2")[D.LABEL]) < 4000


def test_model_file_alone_has_no_data_info(env):
    st, reg, tmp = env
    r = R.run(st, reg, tmp, reg.ref("latest"))
    keys = set(json.loads(open(st.artifact(r["artifacts"]["train.model"])["uri"]).read()))
    assert keys == {"w", "b", "prep"}


def test_trace_matches_independent_dataset_hash(env):
    st, reg, tmp = env
    r = R.run(st, reg, tmp, reg.ref("latest"), args={"seed": 3, "shift": 0.5})
    tr = st.trace(r["artifacts"]["train.model"])
    assert tr["train_data"]["dataset_hash"] == train_hash(3, 4000, 0.5, "v1")
    assert tr["train_data"]["seed"] == 3 and tr["made_by"]["run_id"] == r["run_id"]
    assert tr["made_by"]["image_digest"] == reg.tags["latest"]


def test_cache_trap_latest_tag_reuses_old_data(env):
    """KFP 2.17.2의 키 필드 묶음: 이미지는 이름 문자열만 본다. 같은 태그에 새 이미지를 올려도 캐시가 맞는다."""
    st, reg, tmp = env
    R.run(st, reg, tmp, reg.ref("latest"))
    reg.push("latest", V2)
    r2 = R.run(st, reg, tmp, reg.ref("latest"))
    assert statuses(r2) == ["CACHED"] * 3
    assert R.data_of(st, r2)["dataset_hash"] == train_hash(1, 4000, 1.0, "v1")  # 의도는 v2


def test_digest_ref_or_explicit_version_misses(env):
    st, reg, tmp = env
    d1 = reg.tags["latest"]
    R.run(st, reg, tmp, reg.ref(digest=d1))
    d2 = reg.push("latest", V2)
    r2 = R.run(st, reg, tmp, reg.ref(digest=d2))
    assert statuses(r2) == ["RAN"] * 3
    assert R.data_of(st, r2)["dataset_hash"] == train_hash(1, 4000, 1.0, "v2")
    r3 = R.run(st, reg, tmp, reg.ref("latest"), args={"data_version": "v1"})
    r4 = R.run(st, reg, tmp, reg.ref("latest"), args={"data_version": "v1"})
    assert statuses(r4) == ["CACHED"] * 3  # 키에 든 값만으로 데이터가 정해지면 다시 써도 맞다
    assert R.data_of(st, r4)["dataset_hash"] == R.data_of(st, r3)["dataset_hash"] == train_hash(1, 4000, 1.0, "v1")


def test_env_var_ignored_by_2_17_2_key_but_not_master(tmp_path):
    for policy, want in (("kfp-2.17.2", "CACHED"), ("kfp-master", "RAN")):
        st, reg = ST.Store(), R.Registry()
        reg.push("latest", V1)
        R.run(st, reg, tmp_path, reg.ref("latest"), policy=policy, env={"prepare": {"SOURCE_VERSION": "v1"}})
        r2 = R.run(st, reg, tmp_path, reg.ref("latest"), policy=policy, env={"prepare": {"SOURCE_VERSION": "v2"}})
        assert r2["tasks"][0]["status"] == want, policy


def test_cache_lookup_is_scoped_by_pipeline_and_namespace(env):
    st, reg, tmp = env
    R.run(st, reg, tmp, reg.ref("latest"), namespace="team-a")
    r2 = R.run(st, reg, tmp, reg.ref("latest"), namespace="team-b")
    assert statuses(r2) == ["RAN"] * 3


def test_disabling_prepare_cache_cascades_with_id_key_only(tmp_path):
    for policy, want in (("kfp-2.17.2", ["RAN", "RAN", "RAN"]), ("content", ["RAN", "CACHED", "CACHED"])):
        st, reg = ST.Store(), R.Registry()
        reg.push("latest", V1)
        a = R.run(st, reg, tmp_path, reg.ref("latest"), policy=policy)
        b = R.run(st, reg, tmp_path, reg.ref("latest"), policy=policy, caching={"prepare": False})
        assert statuses(b) == want, policy
        da, db = (st.artifact(r["artifacts"]["prepare.train_data"]) for r in (a, b))
        assert da["id"] != db["id"] and da["sha"] == db["sha"]  # 내용은 같고 ID만 새로


def test_cached_execution_points_to_same_artifact_ids(env):
    st, reg, tmp = env
    a = R.run(st, reg, tmp, reg.ref("latest"))
    b = R.run(st, reg, tmp, reg.ref("latest"))
    assert a["artifacts"] == b["artifacts"]
    tr = st.trace(b["artifacts"]["train.model"])
    assert tr["made_by"]["run_id"] == a["run_id"] and tr["reused_by_runs"] == [b["run_id"]]


def test_sql_and_bfs_lineage_agree(env):
    st, reg, tmp = env
    for i in range(6):
        R.run(st, reg, tmp, reg.ref("latest"), args={"seed": i % 3, "threshold": 0.4 + 0.02 * i})
    for aid in range(1, st.counts()["artifacts"] + 1):
        assert st.ancestors_sql(aid) == st.ancestors_bfs(aid)


def test_descendants_find_every_run_that_used_bad_data(env):
    st, reg, tmp = env
    runs = [R.run(st, reg, tmp, reg.ref("latest"), args=a) for a in ({}, {}, {"threshold": 0.4}, {"seed": 9})]
    _, exes = st.descendants_sql(runs[0]["artifacts"]["prepare.train_data"])
    assert {st.execution(e)["run_id"] for e in exes} == {r["run_id"] for r in runs[:3]}


def test_torch_body_same_accuracy_different_model_file(env):
    st, reg, tmp = env
    a = R.run(st, reg, tmp, reg.ref("latest"), args={"impl": "numpy"})
    b = R.run(st, reg, tmp, reg.ref("latest"), args={"impl": "torch"})
    assert statuses(b) == ["CACHED", "RAN", "RAN"]
    ma, mb = (st.artifact(r["artifacts"]["train.model"]) for r in (a, b))
    wa, ba, _ = S.load_model_weights(ma["uri"])
    wb, bb, _ = S.load_model_weights(mb["uri"])
    assert np.max(np.abs(wa - wb)) < 1e-12 and abs(ba - bb) < 1e-12
    assert R.metrics_of(st, a)["accuracy"] == R.metrics_of(st, b)["accuracy"]
    assert ma["sha"] != mb["sha"]
    assert st.trace(mb["id"])["made_by"]["params"]["impl"] == "torch"


def test_cache_key_fields_follow_kfp_2_17_2():
    key = R.cache_key("kfp-2.17.2", {"x": {"id": 7, "sha": "abc"}}, {"p": 1}, {"o": "system.Dataset"},
                      "r:latest", "d1", "cmd", {"A": "1"})
    assert set(key) == {"inputArtifactNames", "inputParameterValues", "outputArtifactsSpec", "outputParametersSpec",
                        "containerSpec"}
    assert key["inputArtifactNames"] == {"x": {"artifactNames": ["7"]}}
    assert set(key["containerSpec"]) == {"image", "cmdArgs", "pvcNames"}  # env 없음(2.17.2)
    assert "env" in R.cache_key("kfp-master", {}, {}, {}, "r", "d", "c", {"A": "1"})["containerSpec"]
