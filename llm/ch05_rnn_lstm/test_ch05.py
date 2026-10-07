"""5장 검증. `uv run pytest -q llm/ch05_rnn_lstm` 로 실행한다."""

import numpy as np
import torch

import rnn_np as R
import rnn_torch as T


def _rnn_params(rng, d_in=3, H=4, k=2):
    return {
        "W": rng.standard_normal((H, H)) * 0.6,
        "U": rng.standard_normal((H, d_in)) * 0.6,
        "b": rng.standard_normal(H) * 0.1,
        "Wo": rng.standard_normal((k, H)),
        "bo": rng.standard_normal(k) * 0.1,
    }


def _lstm_params(rng, d_in=3, H=4, k=2):
    return {
        "Wx": rng.standard_normal((4 * H, d_in)) * 0.5,
        "Wh": rng.standard_normal((4 * H, H)) * 0.5,
        "b": rng.standard_normal(4 * H) * 0.3,
        "Wo": rng.standard_normal((k, H)),
        "bo": rng.standard_normal(k) * 0.1,
    }


def _check_grads(loss_fn, bptt_fn, params, x, y, tol=1e-6):
    _, g, dx = bptt_fn(params, x, y)
    eps = 1e-6
    for name, P in list(params.items()) + [("x", x)]:
        G = dx if name == "x" else g[name]
        num = np.zeros_like(P)
        for idx in np.ndindex(P.shape):
            old = P[idx]
            P[idx] = old + eps
            lp = loss_fn(params, x, y)
            P[idx] = old - eps
            lm = loss_fn(params, x, y)
            P[idx] = old
            num[idx] = (lp - lm) / (2 * eps)
        rel = np.abs(num - G).max() / max(1e-12, np.abs(num).max())
        assert rel < tol, (name, rel)


def test_rnn_bptt_matches_central_difference():
    rng = np.random.default_rng(0)
    _check_grads(R.rnn_loss, R.rnn_bptt, _rnn_params(rng), rng.standard_normal((6, 3)), 1)


def test_lstm_bptt_matches_central_difference():
    rng = np.random.default_rng(1)
    _check_grads(R.lstm_loss, R.lstm_bptt, _lstm_params(rng), rng.standard_normal((6, 3)), 0)


def test_numpy_matches_torch_manual_and_builtin():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((9, 3))
    p = _rnn_params(rng)
    ref = R.rnn_forward(x, p["W"], p["U"], p["b"])
    xt = torch.tensor(x)
    args = [torch.tensor(p[k]) for k in ("W", "U", "b")]
    np.testing.assert_allclose(T.rnn_manual(xt, *args).numpy(), ref, atol=1e-10)
    np.testing.assert_allclose(T.builtin_rnn(xt, *args).detach().numpy(), ref, atol=1e-10)

    q = _lstm_params(rng)
    ref, _, _ = R.lstm_forward(x, q["Wx"], q["Wh"], q["b"])
    args = [torch.tensor(q[k]) for k in ("Wx", "Wh", "b")]
    np.testing.assert_allclose(T.lstm_manual(xt, *args).numpy(), ref, atol=1e-10)
    np.testing.assert_allclose(T.builtin_lstm(xt, *args).detach().numpy(), ref, atol=1e-10)


def test_gate_order_matters():
    """순서를 바꾸지 않고 그대로 nn.LSTM에 넣으면 결과가 달라진다(맞춘 게 우연이 아니다)."""
    rng = np.random.default_rng(3)
    x = rng.standard_normal((5, 3))
    q = _lstm_params(rng)
    ref, _, _ = R.lstm_forward(x, q["Wx"], q["Wh"], q["b"])
    m = torch.nn.LSTM(3, 4, dtype=torch.float64)
    with torch.no_grad():
        m.weight_ih_l0.copy_(torch.tensor(q["Wx"]))
        m.weight_hh_l0.copy_(torch.tensor(q["Wh"]))
        m.bias_ih_l0.copy_(torch.tensor(q["b"]))
        m.bias_hh_l0.zero_()
    wrong, _ = m(torch.tensor(x))
    assert np.abs(wrong.detach().numpy() - ref).max() > 1e-3


def test_rnn_jacobian_matches_finite_difference():
    """7장 test_ch07.py에서 복사."""
    rng = np.random.default_rng(0)
    d, n = 4, 6
    x, W, U = rng.standard_normal((n, d)), rng.standard_normal((d, d)) * 0.5, rng.standard_normal((d, d)) * 0.5
    h0 = np.zeros(d)
    J_fd = np.empty((d, d))
    eps = 1e-6
    for j in range(d):
        e = np.zeros(d)
        e[j] = eps
        J_fd[:, j] = (R.rnn_forward(x, W, U, h0=h0 + e)[-1] - R.rnn_forward(x, W, U, h0=h0 - e)[-1]) / (2 * eps)
    assert abs(np.linalg.norm(J_fd, 2) - R.rnn_state_jacobian_norms(x, W, U)[-1]) < 1e-6


def test_lstm_state_jacobian_matches_finite_difference():
    rng = np.random.default_rng(4)
    H, n = 3, 5
    x = rng.standard_normal((n, 2))
    Wx, Wh, b = rng.standard_normal((4 * H, 2)) * 0.5, rng.standard_normal((4 * H, H)) * 0.5, rng.standard_normal(4 * H) * 0.5
    eps = 1e-6
    J_fd = np.empty((2 * H, 2 * H))
    for j in range(2 * H):
        e = np.zeros(2 * H)
        e[j] = eps
        hp, cp, _ = R.lstm_forward(x, Wx, Wh, b, h0=e[:H], c0=e[H:])
        hm, cm, _ = R.lstm_forward(x, Wx, Wh, b, h0=-e[:H], c0=-e[H:])
        J_fd[:, j] = (np.concatenate([hp[-1], cp[-1]]) - np.concatenate([hm[-1], cm[-1]])) / (2 * eps)
    cell, state, _ = R.lstm_state_jacobian_norms(x, Wx, Wh, b)
    assert abs(np.linalg.norm(J_fd, 2) - state[-1]) < 1e-6
    assert abs(np.linalg.norm(J_fd[H:, H:], 2) - cell[-1]) < 1e-6


def test_constant_error_carousel():
    """forget 게이트가 1로 열리고 h에서 게이트로 돌아오는 길이 없으면 dc_t/dc_0 = I (Hochreiter·Schmidhuber의 CEC)."""
    H, n = 3, 50
    x = np.random.default_rng(5).standard_normal((n, 2))
    Wx, Wh = np.zeros((4 * H, 2)), np.zeros((4 * H, H))
    b = np.zeros(4 * H)
    b[H : 2 * H] = 40.0  # f = sigmoid(40) = 1 (float64에서)
    cell, _, f_prod = R.lstm_state_jacobian_norms(x, Wx, Wh, b)
    assert abs(cell[-1] - 1.0) < 1e-12 and np.allclose(f_prod, 1.0)


def test_clip_by_norm_keeps_direction():
    rng = np.random.default_rng(6)
    g = [rng.standard_normal((3, 3)) * 10, rng.standard_normal(3) * 10]
    out = R.clip_by_norm(g, 1.0)
    assert abs(R.global_norm(out) - 1.0) < 1e-12
    flat, flat_out = np.concatenate([a.ravel() for a in g]), np.concatenate([a.ravel() for a in out])
    cos = flat @ flat_out / (np.linalg.norm(flat) * np.linalg.norm(flat_out))
    assert abs(cos - 1.0) < 1e-12
    small = [a * 1e-3 for a in g]
    assert all(np.array_equal(a, b) for a, b in zip(R.clip_by_norm(small, 1.0), small))  # 문턱 아래면 그대로


def test_elementwise_clip_can_turn_direction():
    """반례: (10, 0.1)을 원소별로 1에서 자르면 (1, 0.1). 방향이 약 5.1도 틀어진다."""
    g = [np.array([10.0, 0.1])]
    out = R.clip_elementwise(g, 1.0)[0]
    np.testing.assert_allclose(out, [1.0, 0.1])
    cos = g[0] @ out / (np.linalg.norm(g[0]) * np.linalg.norm(out))
    assert np.degrees(np.arccos(cos)) > 5.0
    norm_out = R.clip_by_norm(g, 1.0)[0]
    assert abs(norm_out @ g[0] / (np.linalg.norm(norm_out) * np.linalg.norm(g[0])) - 1.0) < 1e-12


def test_batched_forward_matches_one_sequence_at_a_time():
    rng = np.random.default_rng(7)
    B, n, d, H = 3, 6, 2, 4
    x = rng.standard_normal((B, n, d))
    W, U = rng.standard_normal((H, H)) * 0.5, rng.standard_normal((H, d)) * 0.5
    Wx, Wh, b = rng.standard_normal((4 * H, d)) * 0.5, rng.standard_normal((4 * H, H)) * 0.5, rng.standard_normal(4 * H) * 0.5
    hb, lb = R.rnn_forward_batched(x, W, U), R.lstm_forward_batched(x, Wx, Wh, b)
    for k in range(B):
        np.testing.assert_allclose(hb[k], R.rnn_forward(x[k], W, U)[-1], atol=1e-12)
        np.testing.assert_allclose(lb[k], R.lstm_forward(x[k], Wx, Wh, b)[0][-1], atol=1e-12)


def test_hand_example_numbers():
    """첫 화면에 쓴 손 계산 숫자: 0.8^10 = 0.107, 0.8^50 = 1.4e-5, sigmoid(3) = 0.953, 0.953^50 = 0.09."""
    assert round(0.8**10, 3) == 0.107
    assert f"{0.8**50:.1e}" == "1.4e-05"
    s3 = float(R.sigmoid(3.0))
    assert round(s3, 3) == 0.953
    assert round(s3**50, 2) == 0.09
    assert f"{0.5**50:.1e}" == "8.9e-16"
