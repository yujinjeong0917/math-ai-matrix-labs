"""1장 검증. `uv run pytest -q pipelines/ch01_reproduce` 로 실행한다.

시드 없는 실행의 정확도가 서로 다르다는 테스트는 일부러 두지 않았다. 시험 400행에서는
우연히 같은 정확도가 나올 수 있어, 그 테스트 자체가 흔들리는 테스트가 되기 때문이다.
대신 시험 행 목록이 다르다는 것(사실상 항상 참)과, 기록해 둔 엔트로피로 다시 낼 수 있다는 것을 본다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pipelines_ch01_data as dat  # noqa: E402
import pipelines_ch01_model_np as mnp  # noqa: E402
import pipelines_ch01_model_torch as mtt  # noqa: E402
import pipelines_ch01_repro as rp  # noqa: E402

torch.set_num_threads(2)


def test_same_seed_same_metrics():
    a, b = mnp.train({"seed": 7}), mnp.train({"seed": 7})
    assert a["accuracy"] == b["accuracy"]
    assert a["weights_fp"] == b["weights_fp"] and a["proba_fp"] == b["proba_fp"]
    assert np.array_equal(a["_w"], b["_w"])


def test_different_seed_different_split():
    a, b = mnp.prepare({"seed": 0}), mnp.prepare({"seed": 1})
    assert not np.array_equal(a["test_idx"], b["test_idx"])


def test_unseeded_runs_draw_different_splits():
    a, b = mnp.prepare({"seed": None}), mnp.prepare({"seed": None})
    assert a["entropy"] != b["entropy"]
    assert not np.array_equal(a["test_idx"], b["test_idx"])


def test_recorded_entropy_replays_unseeded_run():
    first = mnp.train({"seed": None})
    again = mnp.train({"entropy": int(first["entropy"])})
    assert again["accuracy"] == first["accuracy"]
    assert again["weights_fp"] == first["weights_fp"]


def test_split_is_disjoint_and_complete():
    tr, te = mnp.split(2000, 0.2, np.random.default_rng(0))
    assert len(te) == 400 and len(tr) == 1600
    assert set(tr).isdisjoint(te) and len(set(tr) | set(te)) == 2000


def test_dataset_is_fixed_and_schema_v2_changes_fee_only():
    X1, y1 = dat.make_dataset()
    X1b, y1b = dat.make_dataset()
    X2, y2 = dat.make_dataset(schema_version=2)
    assert np.array_equal(X1, X1b) and np.array_equal(y1, y1b)
    fee = dat.NUMERIC.index("monthly_fee")
    np.testing.assert_allclose(X2[:, fee] * 1000.0, X1[:, fee])
    others = [j for j in range(X1.shape[1]) if j != fee]
    assert np.array_equal(X1[:, others], X2[:, others]) and np.array_equal(y1, y2)


def test_numpy_matches_torch_float64():
    s = mnp.prepare({"seed": 3})
    c = s["cfg"]
    w1, b1 = mnp.fit_sgd(s["X_tr"], s["y_tr"], s["w0"], s["b0"], s["orders"], c["batch"], c["lr"])
    w2, b2 = mtt.fit_sgd(s["X_tr"], s["y_tr"], s["w0"], s["b0"], s["orders"], c["batch"], c["lr"])
    np.testing.assert_allclose(w1, w2, atol=1e-12)
    assert abs(b1 - b2) < 1e-12


def test_torch_manual_seed_split_is_reproducible():
    a = mtt.seeded_split(2000, 0.2, 11)
    b = mtt.seeded_split(2000, 0.2, 11)
    assert np.array_equal(a[1], b[1])


def test_env_snapshot_keys_and_no_personal_info():
    snap = rp.env_snapshot()
    assert set(snap) == set(rp.ENV_KEYS)
    text = repr(snap)
    home = os.path.expanduser("~")
    assert home not in text
    user = os.environ.get("USER") or os.environ.get("USERNAME")
    if user and len(user) > 3:
        assert user not in text


def test_env_diff_reports_only_changes():
    a = {"python": "3.12.13", "numpy": "2.3.3"}
    b = {"python": "3.12.13", "numpy": "2.3.4"}
    assert rp.env_diff(a, b) == {"numpy": ("2.3.3", "2.3.4")}


def test_float_addition_is_not_associative():
    a, b, c = np.float32(1e8), np.float32(-1e8), np.float32(1.0)
    assert (a + b) + c == np.float32(1.0)
    assert a + (b + c) == np.float32(0.0)
    assert (0.1 + 0.2) + 0.3 != 0.1 + (0.2 + 0.3)


def test_sequential_float32_sum_depends_on_order():
    x = np.random.default_rng(0).uniform(0.0, 1.0, 200_000).astype(np.float32)
    xs = x[np.random.default_rng(1).permutation(len(x))]
    assert rp.sequential_sum(x) != rp.sequential_sum(xs)
    exact = rp.exact_sum(x)
    # NumPy의 부분 쌍 합(pairwise)은 한 개씩 더하기보다 정확한 값에 가깝다
    assert abs(float(np.sum(x)) - exact) < abs(float(rp.sequential_sum(x)) - exact)
