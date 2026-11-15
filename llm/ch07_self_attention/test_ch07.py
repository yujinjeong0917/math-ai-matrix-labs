"""7장 검증. `uv run pytest -q` 로 실행한다."""

import numpy as np
import torch

import attention_np as anp
import attention_torch as atn


def _inputs(n=7, d=8, seed=0):
    # 시드 고정: 매 테스트가 같은 입력을 받게 한다. W는 1/sqrt(d)로 줄여 점수 크기를 안정적으로 유지
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    W = [rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(3)]
    return x, W


# 확인: NumPy 구현, 직접 짠 torch 구현, torch 내장 커널이 causal 유무 모두에서 같은 값을 낸다
def test_numpy_matches_torch_manual_and_builtin():
    x, W = _inputs()
    for causal in (False, True):
        ref = anp.self_attention(x, *W, causal=causal)
        xt, Wt = torch.tensor(x), [torch.tensor(w) for w in W]
        manual = atn.self_attention(xt, *Wt, causal=causal).numpy()
        builtin = atn.self_attention_builtin(xt.float(), *[w.float() for w in Wt], causal=causal).numpy()
        # torch.tensor(x)는 float64라 NumPy와 1e-12까지 일치해야 한다
        np.testing.assert_allclose(manual, ref, atol=1e-12)
        # 내장 커널은 float32로 돌려 허용 오차를 1e-5로 넓힌다(부동소수점 정밀도 차이)
        np.testing.assert_allclose(builtin, ref, atol=1e-5)


def test_attention_weights_are_distributions_and_mask_is_exact():
    x, W = _inputs()
    _, A = anp.self_attention(x, *W, causal=True, return_weights=True)
    # 행 합이 1: 각 행이 확률분포
    np.testing.assert_allclose(A.sum(-1), 1.0, atol=1e-12)
    # -inf 마스크라서 미래 위치 가중치가 '작다'가 아니라 정확히 0.0이어야 한다(== 비교)
    assert np.all(A[np.triu_indices(len(x), 1)] == 0.0)


def test_causal_output_ignores_future_tokens():
    x, W = _inputs()
    # 기준 출력. 아래에서 인덱스 4 이후 입력만 바꿔 비교한다
    y = anp.self_attention(x, *W, causal=True)
    x2 = x.copy()
    x2[4:] += 10.0  # 미래 토큰을 크게 바꿔도
    y2 = anp.self_attention(x2, *W, causal=True)
    # 위치 0..3은 위치 4 이후를 볼 수 없으므로 불변이어야 한다. 뒤 4행(y[4:])은 바뀌어야 위 비교가 의미 있다
    np.testing.assert_allclose(y[:4], y2[:4], atol=1e-12)  # 앞 위치의 출력은 그대로
    assert not np.allclose(y[4:], y2[4:])


def test_without_mask_attention_is_permutation_equivariant():
    """8장으로 넘어가는 질문: 순서를 섞어도 출력이 같은 방식으로 섞일 뿐이다."""
    x, W = _inputs()
    # 행 순열. 위치 정보가 없으니 출력 행도 같은 순열로 섞이기만 한다 (P f(X) = f(P X))
    perm = np.random.default_rng(1).permutation(len(x))
    np.testing.assert_allclose(anp.self_attention(x[perm], *W), anp.self_attention(x, *W)[perm], atol=1e-12)


def test_dot_product_variance_grows_with_d():
    """Vaswani 등(2017) 각주 4: 성분이 평균 0, 분산 1이면 q·k의 분산은 d_k."""
    for d in (16, 256):
        # 점수 분산 / d가 1에 가까워야 한다(표본 400개라 15% 허용)
        var, _, _ = anp.softmax_saturation(d, trials=400)
        assert abs(var / d - 1.0) < 0.15
        # sqrt(d)로 나누면 분산이 d와 무관하게 1
        var_scaled, _, _ = anp.softmax_saturation(d, trials=400, scale=True)
        assert abs(var_scaled - 1.0) < 0.15


def test_rnn_jacobian_matches_finite_difference():
    # 유한차분(중앙차분, eps=1e-6)으로 구한 dh_n/dh_0의 노름이 해석적 야코비안과 같은지 확인. 가중치를 0.5배로 줄인 설정(모형의 선택)
    rng = np.random.default_rng(0)
    d, n = 4, 6
    x, W, U = rng.standard_normal((n, d)), rng.standard_normal((d, d)) * 0.5, rng.standard_normal((d, d)) * 0.5
    h0 = np.zeros(d)
    J_fd = np.empty((d, d))
    eps = 1e-6
    for j in range(d):
        e = np.zeros(d)
        e[j] = eps
        J_fd[:, j] = (anp.rnn_forward(x, W, U, h0 + e)[-1] - anp.rnn_forward(x, W, U, h0 - e)[-1]) / (2 * eps)
    assert abs(np.linalg.norm(J_fd, 2) - anp.rnn_state_jacobian_norms(x, W, U)[-1]) < 1e-6


def test_flop_counts():
    # 손계산: n=d=1이면 투영 3*2 + 점수 2 + 혼합 2 = 10
    assert anp.attention_flops(1, 1) == 10
    # 손계산: n=3, d=2이면 3 * (8 + 8) = 48
    assert anp.rnn_flops(3, 2) == 3 * 16
    n, d = 2048, 64
    assert anp.attention_flops(n, d) > anp.rnn_flops(n, d)  # Q·K·V 투영(6nd^2)까지 세면 attention이 항상 더 많다
    assert anp.attention_flops(32, 64) > anp.rnn_flops(32, 64)  # n < d 여도 마찬가지
