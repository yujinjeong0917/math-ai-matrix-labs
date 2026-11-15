"""9장 검증. `uv run pytest -q pipelines/ch09_drift` 로 실행한다. 네트워크·Jenkins·쿠버네티스·kfp 없이 돈다."""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

import pipelines_ch09_data as D  # noqa: E402
import pipelines_ch09_drift_np as K  # noqa: E402
import pipelines_ch09_drift_torch as KT  # noqa: E402
import pipelines_ch09_jenkinsfile as J  # noqa: E402
import pipelines_ch09_loop as LP  # noqa: E402
import pipelines_ch09_model_np as M  # noqa: E402

torch.set_num_threads(2)
HERE = Path(__file__).parent

# 8장 pipelines_ch08_data.make_table(seed, n, shift)(source_version v1)로 계산해 둔 값
# (시드, 행 수, shift) -> 8장 생성기(v1)가 만든 표의 해시. 기본값 손잡이가 8장 표와 바이트까지 같은지 확인하는 기준
CH08_V1_HASH = {
    (0, 2000, 0.0): "45f62db19eb02f137deebd02d7654e1cfc52479011d4c94db0c59f4fcdfbfb84",
    (1, 4000, 1.0): "39ea38f03c452426af51dcb3f73e8a63be46eff35747fafab46fcc9aad1fe484",
}


@pytest.mark.parametrize("args", list(CH08_V1_HASH))
def test_default_generator_is_ch08_v1(args):
    assert D.dataset_hash(D.make_table(*args)) == CH08_V1_HASH[args]


def test_knobs_change_only_what_they_say():
    base = D.make_table(5, 3000)
    coef = D.make_table(5, 3000, fee_coef=0.4)
    for k in D.NUMERIC + ["plan", "region"]:
        # fee_coef만 바꾸면 입력 열은 그대로이고 라벨만 달라진다
        assert np.array_equal(base[k], coef[k])  # 관계만 바뀜: 입력은 그대로
    assert not np.array_equal(base[D.LABEL], coef[D.LABEL])
    corr = D.make_table(5, 3000, fee_usage_corr=0.7)
    assert np.array_equal(base["monthly_fee"], corr["monthly_fee"])  # 요금 열은 그대로
    # 상관을 넣어도 사용 시간의 평균·표준편차는 거의 그대로(허용 1.0은 표본 3000행의 변동을 고려한 값). 요금과의 상관은 0.6 넘게
    assert abs(corr["usage_hours"].mean() - 40) < 1.0 and abs(corr["usage_hours"].std() - 12) < 1.0
    assert np.corrcoef(corr["monthly_fee"], corr["usage_hours"])[0, 1] > 0.6
    moved = D.make_table(5, 3000, fee_shift=0.5)
    # fee_shift 0.5 = 0.5 x 15,000 = 7,500원. 자르기·반올림 영향으로 300원까지 허용
    assert abs((moved["monthly_fee"].mean() - base["monthly_fee"].mean()) - 7500) < 300


def test_ks_hand_example_and_brute_force():
    # 손 계산: 누적 비율 차이의 최댓값이 0.4. 아래 brute는 모든 값에서 직접 비교해 구현과 대조
    assert K.ks_stat([3, 4, 5, 6, 7], [5, 6, 6, 7, 8]) == pytest.approx(0.4)
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=37), rng.normal(0.3, 1, size=41)
    grid = np.concatenate([a, b])
    brute = max(abs(np.mean(a <= g) - np.mean(b <= g)) for g in grid)
    assert K.ks_stat(a, b) == pytest.approx(brute)


def test_ks_false_alarm_near_alpha_on_continuous_column():
    rng = np.random.default_rng(1)
    hits = [K.ks_pvalue(K.ks_stat(rng.uniform(size=150), rng.uniform(size=150)), 150, 150) < 0.05 for _ in range(400)]
    rate = np.mean(hits)
    se = np.sqrt(0.05 * 0.95 / 400)
    # 오경보율이 alpha 근처인지: 위쪽은 표준오차 3배, 아래쪽(근사가 보수적)은 4배까지 허용
    assert rate <= 0.05 + 3 * se and rate >= 0.05 - 4 * se  # 근사라 아래쪽(보수적)은 조금 더 허용


def test_ks_bonferroni_combined_rate_at_most_alpha():
    alarms = [K.ks_check(D.numeric_matrix(D.make_table(500 + s, 200)), D.numeric_matrix(D.make_table(900 + s, 200)))["alarm"]
              for s in range(200)]
    # 본페로니 보정 후 전체 오경보율이 0.05를 넘지 않는지(표준오차 3배 허용)
    assert np.mean(alarms) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 200)


def test_ks_asymptotic_close_to_permutation_on_continuous_column():
    a = D.make_table(11, 300)["discount_rate"]
    b = D.make_table(12, 300, fee_shift=0.0)["discount_rate"] * 1.08
    p_asym = K.ks_pvalue(K.ks_stat(a, b), 300, 300)
    p_perm = K.ks_perm_pvalue(a, b, n_perm=400, seed=0)
    # 점근 p와 순열 p가 0.06 안에서 같으면 연속 열에서는 근사가 쓸 만하다(허용치는 이 시드의 관찰에 맞춘 값)
    assert abs(p_asym - p_perm) < 0.06


def test_large_shift_alarms_both():
    ref = D.numeric_matrix(D.make_table(21, 500))
    cur = D.numeric_matrix(D.make_table(22, 500, fee_shift=0.5))
    assert K.ks_check(ref, cur, names=D.NUMERIC)["worst"] == "monthly_fee"
    assert K.ks_check(ref, cur)["alarm"]
    assert K.mmd_permutation_test(ref, cur, n_perm=100, seed=0)["p"] < 0.05


def test_mmd_vectorized_equals_loop_and_torch():
    z = K.standardize_pooled(D.numeric_matrix(D.make_table(31, 40)), D.numeric_matrix(D.make_table(32, 50, fee_shift=0.5)))
    x, y = z[:40], z[40:]
    sigma = K.median_bandwidth(z)
    v = K.mmd2_unbiased(x, y, sigma)
    # 벡터화 구현, 이중 루프, PyTorch, 블록 구현이 1e-12 안에서 같은 값을 내는지
    assert v == pytest.approx(K.mmd2_loop(x, y, sigma), abs=1e-12)
    assert v == pytest.approx(KT.mmd2_unbiased(x, y, sigma), abs=1e-12)
    assert v == pytest.approx(K.mmd2_blockwise(x, y, sigma, block=16), abs=1e-12)


def test_permutation_matmul_observed_equals_direct():
    ref = D.numeric_matrix(D.make_table(41, 120))
    cur = D.numeric_matrix(D.make_table(42, 90, fee_shift=0.25))
    out = K.mmd_permutation_test(ref, cur, n_perm=20, seed=0)
    z = K.standardize_pooled(ref, cur)
    assert out["mmd2"] == pytest.approx(K.mmd2_unbiased(z[:120], z[120:], out["sigma"]), abs=1e-12)
    # 순열 20번이면 p의 최솟값은 1/21
    assert 1 / 21 <= out["p"] <= 1.0


def test_mmd_false_alarm_rate_independent_pairs():
    alarms = [K.mmd_permutation_test(D.numeric_matrix(D.make_table(700 + s, 150)), D.numeric_matrix(D.make_table(800 + s, 150)),
                                     n_perm=100, seed=s)["p"] < 0.05 for s in range(150)]
    assert np.mean(alarms) <= 0.05 + 3 * np.sqrt(0.05 * 0.95 / 150)


def test_label_only_change_is_invisible_to_input_checks():
    ref_t, cur_t = D.make_table(51, 500), D.make_table(51, 500, fee_coef=0.0)
    ref, cur = D.numeric_matrix(ref_t), D.numeric_matrix(cur_t)
    assert np.array_equal(ref, cur)  # 같은 고객, 같은 입력
    # 입력이 비트까지 같으니 KS는 아무것도 못 본다(p = 1). 그런데 라벨 관계가 바뀌어 정확도는 0.05 넘게 떨어진다
    assert K.ks_check(ref, cur)["min_p"] == 1.0
    model = M.train(D.make_table(0, 2000))
    assert M.accuracy(model, cur_t) < M.accuracy(model, ref_t) - 0.05


@pytest.fixture(scope="module")
def setup():
    train = D.make_table(0, 2000)
    model = M.train(train)
    # 14개 배치: 매 배치 요금 평균이 오르고 요금 계수가 줄어든다
    batches = [D.make_table(100 + t, 300, fee_shift=0.04 * t, fee_coef=1.6 - 0.04 * t) for t in range(14)]
    return train, model, batches


def test_rollback_restores_previous_digest(setup):
    train, model, batches = setup
    log = LP.run_loop(batches, model, D.numeric_matrix(train), policy="naive", label_delay=3, pause_after_rollback=True)
    assert log["decisions"] and log["decisions"][0]["verdict"] == "rollback"
    d = log["decisions"][0]["t"]
    assert log["serving"][d + 1] == LP.digest(model)  # 판정 다음 배치부터 이전 다이제스트
    assert len(log["runs"]) == 1  # 멈춤 뒤에는 재학습을 내지 않는다
    # 나쁜 모델은 정답이 도착할 때까지(label_delay + 1 배치) 서빙했다
    bad = log["runs"][0]["digest"]
    # 나쁜 모델은 정답이 도착할 때까지 label_delay + 1 = 4배치 서빙한다. 비교를 digest 앞 19자("sha256:" 포함)로 한다
    assert sum(s[:19] == bad for s in log["serving"]) == 3 + 1


def test_labeled_retrain_is_not_rolled_back(setup):
    train, model, batches = setup
    log = LP.run_loop(batches, model, D.numeric_matrix(train), policy="labeled", label_delay=3)
    # labeled 정책은 정답이 도착한 배치(t - 3 이하)만 학습에 쓴다
    assert log["runs"] and all(r["train_batches"][-1] <= r["t"] - 3 for r in log["runs"])
    assert all(dec["verdict"] != "rollback" for dec in log["decisions"])


def test_no_monitor_serves_one_digest(setup):
    train, model, batches = setup
    log = LP.run_loop(batches, model, D.numeric_matrix(train), monitor=False)
    assert len(set(log["serving"])) == 1 and not log["alarms"]


def test_jenkinsfile_rules():
    text = (HERE / "Jenkinsfile").read_text()
    assert J.lint(text) == []
    assert [n for n, _ in J.stages(text)] == ["Drift check", "Submit retrain run", "Gate", "Rollback"]
    assert any("태그" in p for p in J.lint(text.replace('"$SERVING@$(cat results/previous_digest.txt)"', '"$SERVING:latest"')))
    assert any("H" in p for p in J.lint(text.replace("cron('H * * * *')", "cron('0 * * * *')")))
    assert any("학습" in p for p in J.lint(text.replace("python ci/submit_retrain.py", "python -c 'M.train(x)'; python ci/submit_retrain.py")))


def test_hand_numbers_on_the_page():
    """웹 챕터의 손 계산 숫자(연습 1~3, 문턱, 29번 배치의 요금 이동)를 다시 계산한다."""
    import math
    assert K.ks_stat([3, 4, 5, 6, 7], [4, 5, 6, 7, 8]) == pytest.approx(0.2)
    # 웹 챕터의 손 계산: 본페로니(열 6개)와 꼬리 첫 항 근사로 구한 lambda. 기준 2000행 vs 배치 500행이면 D 문턱 0.0828
    lam = math.sqrt(-0.5 * math.log(0.05 / 6 / 2))
    assert round(lam, 4) == 1.6554
    assert round(lam * math.sqrt(2500 / 1_000_000), 4) == 0.0828
    assert round(math.sqrt(1000 / 250_000), 4) == 0.0632
    assert round(lam * math.sqrt(1000 / 250_000), 4) == 0.1047
    # 열 k개를 보정 없이 각각 alpha=0.05로 검정하면(열끼리 독립이라 가정) 한 배치에서 헛경보가 한 번이라도 날 확률은 1 - 0.95^k. 30배치의 기대 헛경보 배치 수는 k=10이면 12.04, k=6이면 7.9472
    assert round(30 * (1 - 0.95 ** 10), 2) == 12.04
    assert round(30 * (1 - 0.95 ** 6), 4) == 7.9472
    # 29번 배치의 요금 이동 = 0.04 x 29 x 15,000 = 17,400원
    assert round(0.04 * 29 * D.FEE_STD) == 17_400
