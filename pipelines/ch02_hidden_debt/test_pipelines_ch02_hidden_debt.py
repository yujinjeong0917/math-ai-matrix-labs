"""2장 검증. `uv run pytest -q pipelines/ch02_hidden_debt` 로 실행한다."""

import numpy as np

import pipelines_ch02_checks as chk
import pipelines_ch02_data as dat
import pipelines_ch02_model_np as mnp
import pipelines_ch02_model_torch as mtt

V2_CODEBOOK = dat.UPSTREAM[2]["plan_codebook"]


def _setup(seed=0, plan_probs=(0.5, 0.3, 0.2), n_train=3000, n_serve=1500):
    tr = dat.make_customers(seed, n_train, plan_probs)
    sv = dat.make_customers(1000 + seed, n_serve, plan_probs)
    train_batch = dat.emit_batch(tr, 1)
    return tr, sv, train_batch, mnp.train(train_batch, tr["churned"])


# Infra 2 · 모델 명세 단위 테스트: NumPy 경사와 PyTorch autograd가 같은 가중치를 낸다
def test_numpy_matches_torch():
    tr, _, batch, _ = _setup()
    X = mnp.transform(batch, mnp.fit_preprocessor(batch))
    w1, b1 = mnp.fit_logistic(X, tr["churned"], steps=200)
    w2, b2 = mtt.fit_logistic(X, tr["churned"], steps=200)
    np.testing.assert_allclose(w1, w2, atol=1e-12)
    assert abs(b1 - b2) < 1e-12


# Infra 1 · 학습 재현: 같은 시드, 같은 데이터면 가중치가 비트까지 같다
def test_same_seed_same_model():
    _, _, _, (p1, w1, b1) = _setup(seed=3)
    _, _, _, (p2, w2, b2) = _setup(seed=3)
    assert p1 == p2 and np.array_equal(w1, w2) and b1 == b2


# Data 7 · 입력 피처 코드 테스트: 학습 표에서 표준화 결과는 평균 0, 표준편차 1, 원-핫은 행마다 1
def test_transform_standardizes_and_onehots():
    _, _, batch, (prep, _, _) = _setup()
    X = mnp.transform(batch, prep)
    np.testing.assert_allclose(X[:, :6].mean(0), 0.0, atol=1e-9)
    np.testing.assert_allclose(X[:, :6].std(0), 1.0, atol=1e-9)
    np.testing.assert_array_equal(X[:, 6:9].sum(1), 1.0)
    np.testing.assert_array_equal(X[:, 9:13].sum(1), 1.0)


# 조용한 실패: v2 표는 예외 없이 예측되지만 정확도가 크게 떨어진다
def test_v2_fails_silently():
    _, sv, _, model = _setup()
    v1 = mnp.accuracy(dat.emit_batch(sv, 1), sv["churned"], model)
    prep, w, b = model
    p = mnp.predict_proba(mnp.transform(dat.emit_batch(sv, 2), prep), w, b)  # 예외가 나면 이 테스트가 실패한다
    v2 = float(np.mean((p >= 0.5) == sv["churned"]))
    assert not np.isnan(p).any()
    assert v1 > 0.78 and v1 - v2 > 0.2


# Data 1 · Monitor 2: 스키마는 v1을 통과시키고 단위가 바뀐 표를 막는다
def test_schema_accepts_v1_and_blocks_unit_change():
    _, sv, _, _ = _setup()
    assert chk.validate(dat.emit_batch(sv, 1)) == []
    errors = chk.validate(dat.emit_batch(sv, 1, fee_divisor=1000.0))
    assert any("monthly_fee" in e for e in errors)


# 값 검사의 한계 1: 코드 순서만 바뀌면 스키마는 통과, 비율이 다르면 분포 비교가 잡는다
def test_code_reorder_passes_schema_but_drift_catches():
    tr, sv, train_batch, _ = _setup()
    serve = dat.emit_batch(sv, 1, plan_codebook=V2_CODEBOOK)
    assert chk.validate(serve) == []
    assert "plan" in chk.drift_alerts(chk.profile(train_batch), serve)


# 값 검사의 한계 2: 요금제 비율이 고르면 어떤 값 검사도 못 잡지만 정확도는 떨어진다
def test_uniform_code_reorder_slips_through_value_checks():
    tr, sv, train_batch, model = _setup(plan_probs=(1 / 3, 1 / 3, 1 / 3))
    serve = dat.emit_batch(sv, 1, plan_codebook=V2_CODEBOOK)
    assert chk.validate(serve) == []
    assert chk.drift_alerts(chk.profile(train_batch), serve) == {}
    clean = mnp.accuracy(dat.emit_batch(sv, 1), sv["churned"], model)
    assert clean - mnp.accuracy(serve, sv["churned"], model) > 0.03


# 뜻이 담긴 계약: 단위가 바뀌면 열 이름이 달라져 막히고, 코드표가 바뀌어도 라벨로 받으니 상관없다
def test_labeled_contract_blocks_unit_and_ignores_codebook():
    tr, sv, _, _ = _setup()
    lab = mnp.labeled_to_codes(dat.emit_labeled_batch(sv), dat.PLAN_LABELS, dat.REGION_LABELS)
    np.testing.assert_array_equal(lab["plan"], dat.emit_batch(sv, 1)["plan"])
    try:
        mnp.labeled_to_codes(dat.emit_labeled_batch(sv, fee_divisor=1000.0), dat.PLAN_LABELS, dat.REGION_LABELS)
        raise AssertionError("단위가 바뀐 표가 계약을 통과했다")
    except mnp.ContractError:
        pass


# Model 5 · Model 6 · Infra 4: 다수 클래스 기준선보다 낫고, 요금제별 조각에서도 0.75 이상
def test_model_beats_baseline_overall_and_by_plan():
    _, sv, _, model = _setup()
    b, y = dat.emit_batch(sv, 1), sv["churned"]
    majority = max(y.mean(), 1 - y.mean())
    assert mnp.accuracy(b, y, model) >= majority + 0.05
    for code in range(3):
        m = b["plan"] == code
        assert mnp.accuracy({k: v[m] for k, v in b.items()}, y[m], model) >= 0.75


# 단위 자체가 문제가 아니라 '학습과 서빙이 다른 단위'가 문제다
def test_consistent_unit_change_is_harmless():
    tr, sv, _, model = _setup()
    model_k = mnp.train(dat.emit_batch(tr, 1, fee_divisor=1000.0), tr["churned"])
    a = mnp.accuracy(dat.emit_batch(sv, 1), sv["churned"], model)
    k = mnp.accuracy(dat.emit_batch(sv, 1, fee_divisor=1000.0), sv["churned"], model_k)
    assert abs(a - k) < 1e-9


# ML Test Score 계산법(Breck 등 2017 §VI.A): 수동 0.5점, 자동 1점, 네 영역 합의 최솟값
def test_ml_test_score_rules():
    s = chk.ml_test_score([("data", "a", "automated"), ("data", "b", "manual"), ("model", "c", "automated"),
                           ("infra", "d", "automated"), ("infra", "e", "automated"), ("monitor", "f", "none")])
    assert s["by_section"] == {"data": 1.5, "model": 1.0, "infra": 2.0, "monitor": 0.0}
    assert s["final"] == 0.0
    assert chk.ml_test_score(chk.THIS_CHAPTER)["final"] == 1.0


# 학습 시간 상한: CI에 넣을 수 있는 크기인지(넉넉하게 5초)
def test_training_time_upper_bound():
    import time

    tr = dat.make_customers(0, 4000)
    t0 = time.perf_counter()
    mnp.train(dat.emit_batch(tr, 1), tr["churned"])
    assert time.perf_counter() - t0 < 5.0
