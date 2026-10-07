"""13장 검증. `uv run pytest -q llm/ch13_scaling` 로 실행한다."""

import numpy as np
import torch
from torch.utils.flop_counter import FlopCounterMode

import ch13_data as cd
import scaling_np as sn
import scaling_torch as st

CFGS = [{"d": 16, "L": 1, "h": 2}, {"d": 48, "L": 3, "h": 4}, {"d": 96, "L": 4, "h": 4}]
TRUE = {"E": 1.8, "A": 40.0, "B": 300.0, "alpha": 0.45, "beta": 0.35}


def _synthetic():
    N = np.repeat([4e3, 1.5e4, 3e4, 9e4, 2e5, 4.5e5], 5)
    D = np.tile([2**17, 2**18, 2**19, 2**20, 2**21], 6).astype(float)
    return N, D, sn.chinchilla_loss(N, D, TRUE)


def test_count_params_matches_torch():
    for cfg in CFGS:
        n, tot = sn.count_params(cfg)
        assert (n, tot) == st.torch_param_counts(st.build(cfg, 0))


def test_exact_forward_flops_match_flop_counter():
    """행렬곱을 손으로 센 순전파 FLOP이 PyTorch FlopCounterMode와 같다."""
    for cfg in CFGS:
        m = st.build(cfg, 0)
        x = torch.randint(0, cd.V, (2, cd.N_CTX))
        with FlopCounterMode(display=False) as fc:
            m(x)
        assert fc.get_total_flops() == 2 * cd.N_CTX * sn.flops_per_token(cfg)["exact_forward"]


def test_6N_is_close_only_when_context_term_is_small():
    small, big = (sn.flops_per_token(c)["ratio_exact_over_6N"] for c in (CFGS[0], CFGS[2]))
    assert small > big > 0.9  # 작은 모델일수록 n_ctx 항과 출력층 몫이 커서 6N에서 멀다


def test_power_law_fit_recovers_exponent():
    N = np.logspace(3, 6, 8)
    L = (8.8e4 / N) ** 0.076
    p = sn.fit_power_law(N, L)
    assert abs(p["alpha"] - 0.076) < 1e-9 and abs(p["Nc"] / 8.8e4 - 1) < 1e-6


def test_numpy_chinchilla_fit_recovers_truth():
    N, D, L = _synthetic()
    p = sn.fit_chinchilla(N, D, L)
    for k in ("alpha", "beta"):
        assert abs(p[k] - TRUE[k]) < 0.01
    assert abs(p["E"] - TRUE["E"]) < 0.01
    assert p["rmse"] < 1e-3


def test_torch_huber_fit_recovers_truth_and_agrees_with_numpy():
    N, D, L = _synthetic()
    pt = st.fit_chinchilla_torch(N, D, L)
    for k in ("alpha", "beta"):
        assert abs(pt[k] - TRUE[k]) < 0.01
    assert abs(pt["E"] - TRUE["E"]) < 0.01


def test_optimal_allocation_minimizes_loss_on_isoflop():
    """식 (4)의 N_opt가 C = 6ND 선 위에서 실제로 손실이 가장 낮은 점이다."""
    C = 1e12
    opt = sn.optimal_allocation(C, TRUE)
    Ns = np.logspace(2, 8, 4001)
    L = sn.chinchilla_loss(Ns, C / (6 * Ns), TRUE)
    assert abs(np.log(Ns[np.argmin(L)] / opt["N_opt"])) < 0.01
    assert abs(6 * opt["N_opt"] * opt["D_opt"] / C - 1) < 1e-9
    assert abs(opt["a"] + opt["b"] - 1) < 1e-12


def test_lr_schedule_shape():
    lrs = [st.lr_at(s, 1.0, 10, 100) for s in range(150)]
    assert lrs[9] == 1.0 and abs(lrs[99] - 0.1) < 1e-3 and lrs[149] == lrs[100]
    assert all(a >= b - 1e-12 for a, b in zip(lrs[10:100], lrs[11:100]))


def test_training_stream_prefix_is_shared():
    """짧은 학습과 긴 학습이 같은 앞부분 토큰을 본다(E4의 중간 읽기 비교가 공정하려면 필요)."""
    a, b = cd.training_stream(300, 5), cd.training_stream(900, 5)
    assert np.array_equal(a, b[:300])


def test_teacher_layout_and_floor():
    """문장은 (열쇠, 열쇠, 값) 묶음이고, 값은 열쇠의 규칙값과 P_RULE 근처 비율로 일치한다."""
    X = cd.sample_sequences(4000, seed=3)
    keys = X[:, 0::3] * cd.V + X[:, 1::3]
    hit = float(np.mean(X[:, 2::3] == cd.VALUE[keys]))
    assert abs(hit - (cd.P_RULE + (1 - cd.P_RULE) / cd.V)) < 0.01
    assert abs(cd.P_KEY.sum() - 1) < 1e-12
    pos = cd.scored_positions()
    assert np.all((pos + 1) % 3 == 2) and len(pos) == cd.GROUPS
    f = cd.entropy_floor()
    assert 0 < f["floor"] < f["uniform"]


def test_fixed_E_fit_recovers_truth():
    N, D, L = _synthetic()
    p = sn.fit_chinchilla(N, D, L, E_fixed=TRUE["E"])
    pt = st.fit_chinchilla_torch(N, D, L, E_fixed=TRUE["E"])
    for q in (p, pt):
        assert abs(q["alpha"] - TRUE["alpha"]) < 0.01 and abs(q["beta"] - TRUE["beta"]) < 0.01
        assert abs(q["E"] - TRUE["E"]) < 1e-9


def test_fixed_seed_training_is_deterministic():
    torch.set_num_threads(1)
    cfg = {"d": 16, "L": 1, "h": 2}
    r1, r2 = (st.train_run(cfg, 2**13, 0, 1e-2) for _ in range(2))
    assert r1["val_loss"] == r2["val_loss"]
