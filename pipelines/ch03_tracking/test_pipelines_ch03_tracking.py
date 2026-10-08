"""3장 검증. `uv run pytest -q pipelines/ch03_tracking` 로 실행한다."""

import json

import numpy as np
import pytest

import pipelines_ch03_data as dat
import pipelines_ch03_hashing as hsh
import pipelines_ch03_model_np as mnp
import pipelines_ch03_model_torch as mtt
import pipelines_ch03_tracking as trk

N_ROWS, N_TRAIN, SPLIT_SEED = 1500, 1000, 7


def make_data(p):
    table = dat.make_table(p["data_seed"], N_ROWS)
    tr, va = dat.split(table, N_TRAIN, p["split_seed"])
    return tr, va, hsh.dataset_hash(table)


def train_fn(tr, va, p):
    return mnp.train_and_eval(tr, va, p["lr"], p["steps"], p["l2"])[1]


P = {"lr": 0.5, "steps": 200, "l2": 0.01, "data_seed": 3, "split_seed": SPLIT_SEED}


def test_hash_stable():
    assert hsh.dataset_hash(dat.make_table(1, 500)) == hsh.dataset_hash(dat.make_table(1, 500))


def test_hash_sensitive():
    t = dat.make_table(1, 500)
    h = hsh.dataset_hash(t)
    t2 = {k: v.copy() for k, v in t.items()}
    t2["age"][0] += 1.0
    assert hsh.dataset_hash(t2) != h
    t3 = {k: v.copy() for k, v in t.items()}
    t3["usage_hours"][7] = np.nextafter(t3["usage_hours"][7], np.inf)
    assert hsh.dataset_hash(t3) != h
    assert hsh.dataset_hash({k: v.astype(np.float32) if v.dtype == np.float64 else v for k, v in t.items()}) != h
    assert hsh.dataset_hash(dat.make_table(2, 500)) != h


def test_hash_ignores_column_order():
    t = dat.make_table(1, 300)
    assert hsh.dataset_hash({k: t[k] for k in reversed(list(t))}) == hsh.dataset_hash(t)


def test_rerun_from_record(tmp_path):
    tr, va, h = make_data(P)
    model, m = mnp.train_and_eval(tr, va, P["lr"], P["steps"], P["l2"])
    rid = trk.log_run(tmp_path, P, m, artifacts={"model": {"w": model["w"], "b": np.array(model["b"])}},
                      data={"train_valid": h}, code=trk.code_identity(), env=trk.env_snapshot())
    rec = trk.load_runs(tmp_path)[0]
    assert rec["run_id"] == rid
    assert trk.rerun_from_record(rec, make_data, train_fn)["val_acc"] == m["val_acc"]
    saved = np.load(tmp_path / rec["artifact_uri"]["model"])
    np.testing.assert_array_equal(saved["w"], model["w"])


def test_rerun_refuses_other_data(tmp_path):
    tr, va, h = make_data(P)
    trk.log_run(tmp_path, P, train_fn(tr, va, P), data={"train_valid": h})
    rec = trk.load_runs(tmp_path)[0]
    with pytest.raises(trk.RecordMismatch):
        trk.rerun_from_record(rec, lambda p: make_data({**p, "data_seed": 4}), train_fn)


def test_log_is_append_only_and_groups(tmp_path):
    ids = []
    for seed in (3, 3, 4):
        p = {**P, "data_seed": seed}
        tr, va, h = make_data(p)
        ids.append(trk.log_run(tmp_path, p, train_fn(tr, va, p), data={"train_valid": h}, code={"source_sha256": "x"}))
    runs = trk.load_runs(tmp_path)
    assert [r["run_id"] for r in runs] == ids and len(set(ids)) == 3
    assert sorted(len(v) for v in trk.comparable_groups(runs).values()) == [1, 2]
    json.loads((tmp_path / "runs.jsonl").read_text().splitlines()[0])


def test_code_identity_tracks_source():
    c = trk.code_identity()
    assert len(c["source_sha256"]) == 64


def test_numpy_matches_torch():
    tr, va, _ = make_data(P)
    prep = mnp.fit_preprocessor(tr)
    X = mnp.transform(tr, prep)
    w1, b1 = mnp.fit_logistic(X, tr[dat.LABEL], lr=0.5, steps=150, l2=0.01)
    w2, b2 = mtt.fit_logistic(X, tr[dat.LABEL], lr=0.5, steps=150, l2=0.01)
    np.testing.assert_allclose(w1, w2, atol=1e-12)
    assert abs(b1 - b2) < 1e-12


def test_csv_roundtrip_looks_same_but_hash_differs():
    t = dat.make_table(1, 200)
    back, _ = hsh.csv_roundtrip(t, digits=6)
    assert max(float(np.max(np.abs(back[k] - t[k]))) for k in t) < 1e-3
    assert hsh.dataset_hash(back) != hsh.dataset_hash(t)
