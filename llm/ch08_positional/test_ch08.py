"""8장 검증. `uv run pytest -q llm/ch08_positional` 로 실행한다."""

import numpy as np
import pytest
import torch

import positional_np as pnp
import positional_torch as ptn


def _inputs(n=7, d=8, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    W = [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(3)]
    return x, W, pnp.learned_pos(16, d, rng)


def test_no_position_no_mask_is_permutation_equivariant():
    """7장 연습 3: 순서를 섞으면 출력도 같은 방식으로 섞일 뿐이다. f(Px) = P f(x)."""
    x, W, _ = _inputs()
    perm = np.random.default_rng(1).permutation(len(x))
    f = lambda z: pnp.attention_with_pos(z, *W, kind="none", causal=False)
    np.testing.assert_allclose(f(x[perm]), f(x)[perm], atol=1e-12)


@pytest.mark.parametrize("kind", ["sin", "learned", "rope", "alibi"])
def test_adding_position_breaks_equivariance(kind):
    x, W, table = _inputs()
    perm = np.random.default_rng(1).permutation(len(x))
    f = lambda z: pnp.attention_with_pos(z, *W, kind=kind, pos_table=table, causal=False)
    assert np.abs(f(x[perm]) - f(x)[perm]).max() > 1e-3


def test_sinusoidal_shift_is_a_linear_map():
    """Vaswani 등 §3.5: 고정된 k에 대해 PE_{pos+k}는 PE_pos의 선형함수. 짝마다 각도 k*freq의 회전이다."""
    n, d, k = 50, 16, 7
    pe = pnp.sinusoidal(n + k, d)
    freq = 10000.0 ** (-np.arange(0, d, 2) / d)
    M = np.zeros((d, d))
    for i, w in enumerate(freq):
        c, s = np.cos(k * w), np.sin(k * w)
        M[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] = [[c, -s], [s, c]]  # [sin, cos] 짝을 k칸 미는 행렬
    np.testing.assert_allclose(pe[:n] @ M, pe[k : n + k], atol=1e-10)


def test_rope_score_depends_only_on_distance():
    """<R_m q, R_n k> 는 (m, n)을 같은 만큼 평행이동해도 변하지 않는다."""
    rng = np.random.default_rng(0)
    q, k = rng.standard_normal((1, 8)), rng.standard_normal((1, 8))
    score = lambda m, n: float((pnp.rope_rotate(q, [m]) @ pnp.rope_rotate(k, [n]).T)[0, 0])
    for m, n in [(5, 2), (40, 3), (0, 9)]:
        for shift in (1, 17, 300):
            assert abs(score(m, n) - score(m + shift, n + shift)) < 1e-10
    assert abs(score(5, 2) - score(5, 3)) > 1e-3  # 거리가 바뀌면 점수도 바뀐다


def test_rope_preserves_norm():
    x = np.random.default_rng(0).standard_normal((10, 8))
    np.testing.assert_allclose(np.linalg.norm(pnp.rope_rotate(x, np.arange(10)), axis=1), np.linalg.norm(x, axis=1), atol=1e-12)


def test_alibi_bias_and_slopes():
    B = pnp.alibi_bias(6, 0.5)
    assert np.all(np.diag(B) == 0.0)
    i, j = np.tril_indices(6)
    np.testing.assert_allclose(B[i, j], -0.5 * (i - j))  # 거리에 정확히 비례
    np.testing.assert_allclose(pnp.alibi_slopes(8), [2.0**-k for k in range(1, 9)])  # 논문: 1/2, 1/4, ..., 1/256
    np.testing.assert_allclose(pnp.alibi_slopes(4), [2.0**-2, 2.0**-4, 2.0**-6, 2.0**-8])


@pytest.mark.parametrize("kind", pnp.KINDS)
def test_numpy_matches_torch(kind):
    x, W, table = _inputs()
    ref = pnp.attention_with_pos(x, *W, kind=kind, pos_table=table, slope=0.25)
    got = ptn.attention_with_pos(torch.tensor(x), *[torch.tensor(w) for w in W], kind=kind, pos_table=torch.tensor(table), slope=0.25)
    np.testing.assert_allclose(got.numpy(), ref, atol=1e-10)


def test_sinusoidal_and_alibi_tables_match_torch():
    np.testing.assert_allclose(ptn.sinusoidal(20, 8, dtype=torch.float64).numpy(), pnp.sinusoidal(20, 8), atol=1e-12)
    np.testing.assert_allclose(ptn.alibi_bias(5, ptn.alibi_slopes(4).double()).numpy()[1], pnp.alibi_bias(5, 2.0**-4))


def test_rope_rotation_matches_complex():
    x = torch.randn(3, 2, 10, 8, dtype=torch.float64)
    pos = torch.arange(10)
    np.testing.assert_allclose(ptn.rope_rotate(x, pos).numpy(), ptn.rope_rotate_complex(x, pos).numpy(), atol=1e-12)
    np.testing.assert_allclose(ptn.rope_rotate(x, pos).numpy(), pnp.rope_rotate(x.numpy(), np.arange(10)), atol=1e-12)


def test_learned_table_has_no_slot_beyond_n_max():
    x, W, table = _inputs(n=20)
    with pytest.raises(ValueError):
        pnp.attention_with_pos(x, *W, kind="learned", pos_table=table)  # 표는 16칸뿐


def test_model_is_causal_for_every_kind():
    for kind in ptn.KINDS:
        torch.manual_seed(0)
        m = ptn.PosAttentionModel(16, 32, 4, kind, n_max=12)
        x = torch.randint(0, 16, (2, 12))
        x2 = x.clone()
        x2[:, 6:] = (x2[:, 6:] + 1) % 16
        torch.testing.assert_close(m(x)[:, :6], m(x2)[:, :6])


def test_extra_cost():
    assert pnp.extra_cost("none", 64, 64, 4) == (0, 0)
    assert pnp.extra_cost("learned", 64, 64, 4, n_max=64) == (4096, 4096)
    assert pnp.extra_cost("rope", 64, 64)[0] == 6 * 64 * 64
    assert pnp.extra_cost("alibi", 64, 64, 4)[0] == 4 * 64 * 64


def test_first_screen_toy_example():
    """첫 화면 숫자: 위치가 없으면 두 문장이 같은 묶음, 위치 (sin 90°i, cos 90°i)를 더하면 다른 묶음."""
    t = pnp.first_screen_toy()
    assert t["position_vectors_deg90"] == [(0, 1), (1, 0), (0, -1)]
    assert t["3 빼기 5"]["with_position"] == [(3, 1), (2, 0), (5, -1)]
    assert t["5 빼기 3"]["with_position"] == [(5, 1), (2, 0), (3, -1)]
    assert t["same_set_without_position"] and not t["same_set_with_position"]
