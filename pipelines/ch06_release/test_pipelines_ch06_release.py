"""6장 검증. `uv run pytest -q pipelines/ch06_release` 로 실행한다. 네트워크·Jenkins·쿠버네티스 없이 돈다."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import pipelines_ch06_data as D
import pipelines_ch06_jenkinsfile as JF
import pipelines_ch06_model_np as M
import pipelines_ch06_model_torch as MT
import pipelines_ch06_schema as SC
from pipelines_ch06_release_gate import DEFAULT_GUARD, OVERALL_ONLY, decide, z_worse
from pipelines_ch06_router import bucket, route
from pipelines_ch06_simulate import metrics_of, simulate_traffic

HERE = Path(__file__).parent
JENKINSFILE = (HERE / "Jenkinsfile").read_text()


@pytest.fixture(scope="module")
def models():
    t = D.make_table(0, 4000)
    tr, va = D.split(t, 3000, 0)
    return tr, va, M.train(tr, D.NUMERIC_V1), M.train(tr, D.NUMERIC_V2)


@pytest.fixture(scope="module")
def traffic(models):
    _, _, v1, v2 = models
    tf = D.serve_view(D.make_table(1000, 10_000))
    bt = D.serve_view(D.make_table(5000, 2000))
    return {
        "c1": M.correct(v1, tf), "c2": M.correct(v2, tf), "seg": tf["region"],
        "users": [f"s0-u{i}" for i in range(10_000)], "base": metrics_of(M.correct(v1, bt), bt["region"]),
    }


# ---------- 라우터: 같은 사용자는 같은 버전 ----------

def test_route_is_deterministic_and_sticky_when_raising_p():
    users = [f"user-{i}" for i in range(3000)]
    assert [route(u, 0.05) for u in users] == [route(u, 0.05) for u in users]
    a = [route(u, 0.05) for u in users]
    b = [route(u, 0.2) for u in users]
    assert not any(x == "v2" and y == "v1" for x, y in zip(a, b))  # 비율을 올려도 v2였던 사람은 v2로 남는다
    share = np.mean([y == "v2" for y in b])
    assert 0.18 < share < 0.22


def test_bucket_does_not_depend_on_python_hash_seed():
    code = "import sys; sys.path.insert(0, %r); from pipelines_ch06_router import bucket; print(repr(bucket('user-7')))" % str(HERE)
    outs = set()
    for seed in ("0", "1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outs.add(subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True).stdout)
    assert outs == {repr(bucket("user-7")) + "\n"}


# ---------- 배포 판정 ----------

def test_gate_holds_below_min_n_and_rolls_back_when_guard_is_violated():
    v1 = {"n": 5000, "errors": 1000}
    assert decide(v1, {"n": 49, "errors": 49}, DEFAULT_GUARD) == "hold"
    assert decide(v1, {"n": 200, "errors": 80}, DEFAULT_GUARD) == "rollback"  # 0.40 vs 0.20
    assert decide(v1, {"n": 200, "errors": 40}, DEFAULT_GUARD) == "hold"
    assert decide(v1, {"n": 400, "errors": 80}, DEFAULT_GUARD) == "promote"


def test_segment_guard_catches_what_the_overall_rate_hides():
    seg1 = {0: {"n": 4000, "errors": 800}, 2: {"n": 1000, "errors": 200}}
    seg2 = {0: {"n": 340, "errors": 51}, 2: {"n": 60, "errors": 30}}  # 구간 0은 더 좋고 구간 2만 크게 나쁘다
    m1 = {"n": 5000, "errors": 1000, "seg": seg1}
    m2 = {"n": 400, "errors": 81, "seg": seg2}
    assert z_worse(m1, m2) < 3.0
    assert decide(m1, m2, OVERALL_ONLY) == "promote"
    assert decide(m1, m2, DEFAULT_GUARD) == "rollback"


# ---------- 데이터: 오프라인에선 좋고 운영의 한 구간에선 나쁘다 ----------

def test_skew_only_touches_busan():
    t = D.make_table(3, 500)
    s = D.serve_view(t)
    m = t["region"] == D.SKEW_REGION
    assert np.allclose(s["usage_drop"][m], t["usage_drop"][m] * 100)
    assert np.array_equal(s["usage_drop"][~m], t["usage_drop"][~m])


def test_v2_wins_offline_but_loses_busan_online(models):
    _, va, v1, v2 = models
    assert M.correct(v2, va).mean() > M.correct(v1, va).mean()
    tf = D.serve_view(D.make_table(1000, 10_000))
    m = tf["region"] == D.SKEW_REGION
    assert M.correct(v2, tf)[m].mean() < M.correct(v1, tf)[m].mean() - 0.2
    assert M.correct(v2, tf)[~m].mean() > M.correct(v1, tf)[~m].mean()


# ---------- 시뮬레이터 ----------

def test_harm_is_exact_counterfactual(traffic):
    T = traffic
    r = simulate_traffic(T["c1"], T["c2"], T["users"], T["seg"], "full", DEFAULT_GUARD, baseline=T["base"])
    served = r["served_v2"]
    assert r["harm"] == int(np.sum(T["c1"][:served] - T["c2"][:served]))
    assert sum(r["harm_by_segment"].values()) == r["harm"]


def test_identical_model_has_zero_harm(traffic):
    T = traffic
    r = simulate_traffic(T["c1"], T["c1"], T["users"], T["seg"], "canary", DEFAULT_GUARD, p=0.2)
    assert r["harm"] == 0


def test_bluegreen_switch_back_serves_v1_from_the_next_request(traffic):
    T = traffic
    r = simulate_traffic(T["c1"], T["c2"], T["users"], T["seg"], "bluegreen", DEFAULT_GUARD, baseline=T["base"])
    assert r["decision"] == "rollback"
    assert r["served_v2"] == r["decided_at"]  # 결정한 요청까지만 v2, 그다음부터 v1


def test_full_replacement_keeps_serving_v2_during_redeploy(traffic):
    T = traffic
    bg = simulate_traffic(T["c1"], T["c2"], T["users"], T["seg"], "bluegreen", DEFAULT_GUARD, baseline=T["base"])
    full = simulate_traffic(T["c1"], T["c2"], T["users"], T["seg"], "full", DEFAULT_GUARD, baseline=T["base"],
                            rollback_delay=500)
    assert full["decided_at"] == bg["decided_at"]
    assert full["served_v2"] == bg["served_v2"] + 500


def test_label_delay_postpones_the_decision(traffic):
    T = traffic
    a = simulate_traffic(T["c1"], T["c2"], T["users"], T["seg"], "bluegreen", DEFAULT_GUARD, baseline=T["base"])
    b = simulate_traffic(T["c1"], T["c2"], T["users"], T["seg"], "bluegreen", DEFAULT_GUARD, baseline=T["base"],
                         label_delay=1000)
    assert b["decided_at"] == a["decided_at"] + 1000


# ---------- 되돌리기와 공유 데이터 ----------

def test_in_place_migration_breaks_rollback_but_expand_does_not(models):
    _, _, v1, v2 = models
    shared = D.serve_view(D.make_table(9000, 2000), skew=False)
    bad = SC.rollback_accuracy(v1, v2, shared, "in_place")
    good = SC.rollback_accuracy(v1, v2, shared, "expand")
    assert bad["v2_after_deploy"] == good["v2_after_deploy"]
    assert good["v1_after_rollback"] == good["v1_before"]
    assert bad["v1_after_rollback"] < bad["v1_before"] - 0.2


# ---------- NumPy와 PyTorch ----------

def test_torch_fit_matches_numpy(models):
    tr, _, v1, _ = models
    X = M.transform(tr, v1["prep"])
    w, b = MT.fit_logistic(X, tr[D.LABEL])
    assert np.max(np.abs(w - v1["w"])) < 1e-12
    assert abs(b - v1["b"]) < 1e-12


# ---------- Jenkinsfile ----------

def test_jenkinsfile_stage_order_and_lint():
    assert [n for n, _ in JF.stages(JENKINSFILE)] == ["Offline gate", "Canary 5%", "Approve 100%", "Promote 100%"]
    assert JF.lint(JENKINSFILE) == []


def test_lint_catches_open_approval_and_early_promotion():
    no_submitter = JENKINSFILE.replace("submitter 'ml-release-owners'", "")
    assert any("submitter" in p for p in JF.lint(no_submitter))
    early = JENKINSFILE.replace("sh 'python deploy/set_weight.py --v2 0.05'", "sh 'python deploy/set_weight.py --v2 1.0'")
    assert any("승인 전에 100%" in p for p in JF.lint(early))
    no_revert = JENKINSFILE.replace("post { failure { sh 'python deploy/set_weight.py --v2 0.0' } }", "")
    assert any("되돌리지" in p for p in JF.lint(no_revert))
