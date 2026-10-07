"""11장 검증. `uv run pytest -q llm/ch11_decoding` 로 실행한다."""

import numpy as np
import pytest
import torch

import decoding_np as dn
import decoding_torch as dt


def _logits(B=6, V=30, seed=0):
    return np.random.default_rng(seed).standard_normal((B, V)) * 2.0  # 연속값이라 동점이 없다


def _bigram_model(V=12, seed=0):
    """테스트용 작은 모델: 마지막 토큰만 보는 로짓 표."""
    W = np.random.default_rng(seed).standard_normal((V, V)) * 2.0
    return lambda ids: W[np.atleast_2d(ids)[:, -1]]


# ---------------------------------------------------------------- 첫 화면 손 계산


def test_first_screen_hand_example():
    """첫 화면 예: 확률 0.5, 0.3, 0.15, 0.05에서 p=0.9면 세 개가 남고(합 0.95) 다시 나누면 약 0.53, 0.32, 0.16."""
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    z = dn.top_p_filter(np.log(probs), 0.9)
    kept = np.isfinite(z)
    assert kept.tolist() == [True, True, True, False]
    assert probs[kept].sum() == pytest.approx(0.95)
    renorm = dn.softmax(z)
    np.testing.assert_allclose(renorm[:3], [0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95])
    assert [round(x, 2) for x in renorm[:3]] == [0.53, 0.32, 0.16]
    assert renorm[3] == 0.0
    # 탐욕은 늘 첫 번째(0.5)만 고른다
    assert int(np.argmax(probs)) == 0


# ---------------------------------------------------------------- 필터의 성질


def test_low_temperature_sampling_equals_argmax():
    z = _logits()
    rng = np.random.default_rng(0)
    for T in (0, 1e-3):
        for _ in range(20):
            np.testing.assert_array_equal(dn.sample_from_logits(z, rng, T=T), z.argmax(-1))


def test_top_k_keeps_exactly_k():
    z = _logits()
    for k in (1, 5, 10, 29):
        assert np.all(np.isfinite(dn.top_k_filter(z, k)).sum(-1) == k)
        kept = np.isfinite(dn.top_k_filter(z, k))
        top = np.argsort(-z, -1)[:, :k]
        assert np.all(np.take_along_axis(kept, top, -1))


def test_top_p_set_is_minimal_and_reaches_p():
    z = _logits()
    probs = dn.softmax(z)
    for p in (0.3, 0.9, 0.95):
        kept = np.isfinite(dn.top_p_filter(z, p))
        for row in range(len(z)):
            mass = probs[row, kept[row]].sum()
            assert mass >= p
            smallest = probs[row, kept[row]].min()
            assert mass - smallest < p  # 하나라도 빼면 p에 못 미친다 = 가장 작은 집합
            # 남긴 토큰은 모두 버린 토큰보다 확률이 높다
            if (~kept[row]).any():
                assert smallest >= probs[row, ~kept[row]].max()


def test_numpy_and_torch_masks_match():
    z = _logits(B=8, V=40, seed=3)
    zt = torch.tensor(z)
    for T in (0.7, 1.0, 1.5):
        for k, p in ((None, None), (10, None), (40, None), (None, 0.9), (None, 0.95), (10, 0.9)):
            a = np.isfinite(dn.shape_logits(z, T, k, p))
            b = torch.isfinite(dt.shape_logits(zt, T, k, p)).numpy()
            np.testing.assert_array_equal(a, b)
            np.testing.assert_allclose(dn.softmax(dn.shape_logits(z, T, k, p)), torch.softmax(dt.shape_logits(zt, T, k, p), -1).numpy(), atol=1e-12)


def test_temperature_order_changes_nucleus():
    """온도를 먼저 거는지 나중에 거는지에 따라 top-p 집합이 달라진다(그래서 순서를 고정한다)."""
    z = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))
    first_T = np.isfinite(dn.shape_logits(z, T=0.5, p=0.9)).sum()
    first_p = np.isfinite(dn.apply_temperature(dn.top_p_filter(z, 0.9), 0.5)).sum()
    assert (first_T, first_p) == (2, 3)


def test_sampling_never_picks_masked_and_matches_frequencies():
    z = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))
    rng = np.random.default_rng(0)
    draws = np.array([dn.sample_from_logits(z, rng, p=0.9)[0] for _ in range(20000)])
    assert not np.any(draws == 3)
    freq = np.bincount(draws, minlength=4) / len(draws)
    np.testing.assert_allclose(freq[:3], [0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95], atol=0.01)


# ---------------------------------------------------------------- 생성


def test_beam_one_equals_greedy_and_score_is_logprob():
    model = _bigram_model()
    prompt = np.array([3, 7])
    g = dn.greedy(model, prompt[None], 15)[0]
    b, score = dn.beam_search(model, prompt, 15, beam=1)
    np.testing.assert_array_equal(b, g)
    lp = dn.sequence_logprob(model, b[None], len(prompt))
    assert score == pytest.approx(lp.sum())


def test_beam_beats_greedy_on_this_toy_model():
    """이 시드의 작은 모델에서 확인한 것일 뿐, 빔이 늘 탐욕 이상이라는 정리는 아니다(빔도 근사 탐색)."""
    model = _bigram_model(seed=4)
    prompt = np.array([1])
    g = dn.greedy(model, prompt[None], 10)
    _, s4 = dn.beam_search(model, prompt, 10, beam=4)
    assert s4 >= dn.sequence_logprob(model, g, 1).sum() - 1e-12


def test_repetition_stats_hand_example():
    # 1 2 3 4 1 2 3 4: 4-gram 5개 중 고유 4개 → rep4 = 1/5. 2-gram 7개 중 고유 4개 → distinct2 = 4/7
    s = dn.repetition_stats([1, 2, 3, 4, 1, 2, 3, 4])
    assert s["rep4"] == pytest.approx(1 / 5)
    assert s["distinct2"] == pytest.approx(4 / 7)


def test_source_rows_are_distributions():
    src = dn.SecondOrderSource(V=10, seed=0, max_support=6)
    np.testing.assert_allclose(src.P.sum(-1), 1.0)
    X = src.sample(50, 30, np.random.default_rng(0))
    assert np.all(src.token_probs(X, 2) > 0)  # 정답 언어의 표본에는 나올 수 없는 토큰이 없다


def test_numpy_model_slides_window_past_n_ctx():
    """위치 임베딩이 n_ctx칸뿐이라, 더 긴 문맥은 마지막 n_ctx개만 넣어야 한다(흔한 실패 1)."""
    torch.manual_seed(0)
    m = dt.TinyLM(vocab=10, n_ctx=8, d=16, L=1, h=2)
    with pytest.raises((IndexError, RuntimeError)):
        m(torch.zeros(1, 9, dtype=torch.long))
    f = dt.numpy_model(m)
    ids = np.random.default_rng(0).integers(0, 10, (2, 20))
    np.testing.assert_allclose(f(ids), f(ids[:, -8:]))
    out = dn.greedy(f, ids, 5)
    assert out.shape == (2, 25)


def test_exercise_hand_example_and_temperature_numbers():
    """연습 1(0.4, 0.3, 0.2, 0.1에 p=0.75)과 3절의 T=0.5 손 계산."""
    z = dn.top_p_filter(np.log(np.array([0.4, 0.3, 0.2, 0.1])), 0.75)
    np.testing.assert_allclose(dn.softmax(z), [0.4 / 0.9, 0.3 / 0.9, 0.2 / 0.9, 0.0])
    t = dn.softmax(dn.apply_temperature(np.log(np.array([0.5, 0.3, 0.15, 0.05])), 0.5))
    assert [round(x, 3) for x in t] == [0.685, 0.247, 0.062, 0.007]
