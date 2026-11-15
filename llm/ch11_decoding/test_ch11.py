"""11장 검증. `uv run pytest -q llm/ch11_decoding` 로 실행한다."""

import numpy as np
import pytest
import torch

import decoding_np as dn
import decoding_torch as dt


def _logits(B=6, V=30, seed=0):
    # 연속 난수라 로짓에 동점이 없다(top-k의 개수 검사가 정확해진다). 값 크기를 2배로 키워 분포를 뾰족하게
    return np.random.default_rng(seed).standard_normal((B, V)) * 2.0  # 연속값이라 동점이 없다


def _bigram_model(V=12, seed=0):
    """테스트용 작은 모델: 마지막 토큰만 보는 로짓 표."""
    # 마지막 토큰만 보는 (V, V) 로짓 표. 행 i = 토큰 i 다음의 로짓
    W = np.random.default_rng(seed).standard_normal((V, V)) * 2.0
    return lambda ids: W[np.atleast_2d(ids)[:, -1]]


# ---------------------------------------------------------------- 첫 화면 손 계산


def test_first_screen_hand_example():
    """첫 화면 예: 확률 0.5, 0.3, 0.15, 0.05에서 p=0.9면 세 개가 남고(합 0.95) 다시 나누면 약 0.53, 0.32, 0.16."""
    # 첫 화면 예. 확률 0.5, 0.3, 0.15, 0.05
    probs = np.array([0.5, 0.3, 0.15, 0.05])
    # log 확률을 로짓으로 써도 softmax하면 원래 확률. p = 0.9
    z = dn.top_p_filter(np.log(probs), 0.9)
    kept = np.isfinite(z)
    assert kept.tolist() == [True, True, True, False]
    # 0.5 + 0.3 + 0.15 = 0.95 >= 0.9. 두 개(0.8)는 모자라 세 번째까지 포함
    assert probs[kept].sum() == pytest.approx(0.95)
    renorm = dn.softmax(z)
    # 남긴 토큰을 합 1로 다시 정규화: 0.5/0.95 = 0.526, 0.3/0.95 = 0.316, 0.15/0.95 = 0.158
    np.testing.assert_allclose(renorm[:3], [0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95])
    assert [round(x, 2) for x in renorm[:3]] == [0.53, 0.32, 0.16]
    assert renorm[3] == 0.0
    # 탐욕은 늘 첫 번째(0.5)만 고른다
    assert int(np.argmax(probs)) == 0


# ---------------------------------------------------------------- 필터의 성질


def test_low_temperature_sampling_equals_argmax():
    z = _logits()
    rng = np.random.default_rng(0)
    # T = 0과 매우 낮은 T는 모두 argmax와 같아야 한다(여러 번 뽑아도)
    for T in (0, 1e-3):
        for _ in range(20):
            np.testing.assert_array_equal(dn.sample_from_logits(z, rng, T=T), z.argmax(-1))


def test_top_k_keeps_exactly_k():
    z = _logits()
    for k in (1, 5, 10, 29):
        # 남는(유한) 칸이 정확히 k개. 연속 로짓이라 동점이 없는 설정
        assert np.all(np.isfinite(dn.top_k_filter(z, k)).sum(-1) == k)
        kept = np.isfinite(dn.top_k_filter(z, k))
        # 원래 로짓 상위 k개의 위치
        top = np.argsort(-z, -1)[:, :k]
        assert np.all(np.take_along_axis(kept, top, -1))


def test_top_p_set_is_minimal_and_reaches_p():
    z = _logits()
    probs = dn.softmax(z)
    for p in (0.3, 0.9, 0.95):
        kept = np.isfinite(dn.top_p_filter(z, p))
        for row in range(len(z)):
            mass = probs[row, kept[row]].sum()
            # 남긴 질량이 p 이상
            assert mass >= p
            smallest = probs[row, kept[row]].min()
            # 가장 작은 확률 토큰 하나를 빼면 p 미만: 최소 집합
            assert mass - smallest < p  # 하나라도 빼면 p에 못 미친다 = 가장 작은 집합
            # 남긴 토큰은 모두 버린 토큰보다 확률이 높다
            if (~kept[row]).any():
                # 남긴 것 중 최솟값이 버린 것 중 최댓값 이상: 확률 높은 순으로 남았다
                assert smallest >= probs[row, ~kept[row]].max()


def test_numpy_and_torch_masks_match():
    z = _logits(B=8, V=40, seed=3)
    zt = torch.tensor(z)
    for T in (0.7, 1.0, 1.5):
        for k, p in ((None, None), (10, None), (40, None), (None, 0.9), (None, 0.95), (10, 0.9)):
            # NumPy와 torch 필터가 남기는 칸(마스크)이 같은지
            a = np.isfinite(dn.shape_logits(z, T, k, p))
            b = torch.isfinite(dt.shape_logits(zt, T, k, p)).numpy()
            np.testing.assert_array_equal(a, b)
            # 남은 확률분포도 1e-12 이내로 같다
            np.testing.assert_allclose(dn.softmax(dn.shape_logits(z, T, k, p)), torch.softmax(dt.shape_logits(zt, T, k, p), -1).numpy(), atol=1e-12)


def test_temperature_order_changes_nucleus():
    """온도를 먼저 거는지 나중에 거는지에 따라 top-p 집합이 달라진다(그래서 순서를 고정한다)."""
    z = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))
    # 온도 0.5 먼저: 확률이 0.685, 0.247, ... 로 뾰족해져 p = 0.9에 두 개만 필요
    first_T = np.isfinite(dn.shape_logits(z, T=0.5, p=0.9)).sum()
    # top-p 먼저: 0.5 + 0.3 + 0.15 = 0.95로 세 개 남긴 뒤 온도를 걸어도 세 개 유지
    first_p = np.isfinite(dn.apply_temperature(dn.top_p_filter(z, 0.9), 0.5)).sum()
    assert (first_T, first_p) == (2, 3)


def test_sampling_never_picks_masked_and_matches_frequencies():
    z = np.log(np.array([[0.5, 0.3, 0.15, 0.05]]))
    rng = np.random.default_rng(0)
    # 20000번 뽑아 경험 빈도를 구한다. 시드 고정
    draws = np.array([dn.sample_from_logits(z, rng, p=0.9)[0] for _ in range(20000)])
    # 마스크된 토큰(확률 0.05)은 한 번도 안 뽑혀야 한다
    assert not np.any(draws == 3)
    freq = np.bincount(draws, minlength=4) / len(draws)
    # 빈도가 재정규화 확률과 일치(허용 0.01). 표본 20000개의 표준오차는 약 0.0035이므로 3시그마 정도의 여유(근사)
    np.testing.assert_allclose(freq[:3], [0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95], atol=0.01)


# ---------------------------------------------------------------- 생성


def test_beam_one_equals_greedy_and_score_is_logprob():
    model = _bigram_model()
    prompt = np.array([3, 7])
    # 빔 1개는 탐욕과 같은 열을 줘야 한다. prompt[None]: (1, L0)
    g = dn.greedy(model, prompt[None], 15)[0]
    b, score = dn.beam_search(model, prompt, 15, beam=1)
    np.testing.assert_array_equal(b, g)
    lp = dn.sequence_logprob(model, b[None], len(prompt))
    # 빔의 누적 점수 = 그 열을 원래 모델로 잰 로그확률의 합
    assert score == pytest.approx(lp.sum())


def test_beam_beats_greedy_on_this_toy_model():
    """이 시드의 작은 모델에서 확인한 것일 뿐, 빔이 늘 탐욕 이상이라는 정리는 아니다(빔도 근사 탐색)."""
    # 이 시드에서 빔 4가 탐욕보다 점수가 높거나 같음을 확인. 일반 정리가 아니다(docstring 참고)
    model = _bigram_model(seed=4)
    prompt = np.array([1])
    g = dn.greedy(model, prompt[None], 10)
    _, s4 = dn.beam_search(model, prompt, 10, beam=4)
    assert s4 >= dn.sequence_logprob(model, g, 1).sum() - 1e-12


def test_repetition_stats_hand_example():
    # 1 2 3 4 1 2 3 4: 4-gram 5개 중 고유 4개 → rep4 = 1/5. 2-gram 7개 중 고유 4개 → distinct2 = 4/7
    # 길이 8: 4-gram 5개, 2-gram 7개
    s = dn.repetition_stats([1, 2, 3, 4, 1, 2, 3, 4])
    assert s["rep4"] == pytest.approx(1 / 5)
    assert s["distinct2"] == pytest.approx(4 / 7)


def test_source_rows_are_distributions():
    src = dn.SecondOrderSource(V=10, seed=0, max_support=6)
    # 모든 (a, b) 상태의 다음 토큰 분포 합이 1
    np.testing.assert_allclose(src.P.sum(-1), 1.0)
    X = src.sample(50, 30, np.random.default_rng(0))
    # 표본의 모든 토큰 확률이 0보다 크다(정답 언어에서 나올 수 없는 토큰이 표본에 없다)
    assert np.all(src.token_probs(X, 2) > 0)  # 정답 언어의 표본에는 나올 수 없는 토큰이 없다


def test_numpy_model_slides_window_past_n_ctx():
    """위치 임베딩이 n_ctx칸뿐이라, 더 긴 문맥은 마지막 n_ctx개만 넣어야 한다(흔한 실패 1)."""
    torch.manual_seed(0)
    m = dt.TinyLM(vocab=10, n_ctx=8, d=16, L=1, h=2)
    # n_ctx = 8인데 길이 9를 넣으면 위치 표 범위를 넘어 오류(torch 버전에 따라 IndexError 또는 RuntimeError)
    with pytest.raises((IndexError, RuntimeError)):
        m(torch.zeros(1, 9, dtype=torch.long))
    f = dt.numpy_model(m)
    ids = np.random.default_rng(0).integers(0, 10, (2, 20))
    # 길이 20 문맥과 마지막 8개만 준 경우의 로짓이 같다: 슬라이딩 창이 동작
    np.testing.assert_allclose(f(ids), f(ids[:, -8:]))
    out = dn.greedy(f, ids, 5)
    assert out.shape == (2, 25)


def test_exercise_hand_example_and_temperature_numbers():
    """연습 1(0.4, 0.3, 0.2, 0.1에 p=0.75)과 3절의 T=0.5 손 계산."""
    z = dn.top_p_filter(np.log(np.array([0.4, 0.3, 0.2, 0.1])), 0.75)
    # p = 0.75: 0.4 + 0.3 = 0.7 < 0.75이므로 0.2까지 포함해 합 0.9. 재정규화 0.4/0.9 등
    np.testing.assert_allclose(dn.softmax(z), [0.4 / 0.9, 0.3 / 0.9, 0.2 / 0.9, 0.0])
    t = dn.softmax(dn.apply_temperature(np.log(np.array([0.5, 0.3, 0.15, 0.05])), 0.5))
    # T = 0.5이면 확률을 제곱한 뒤 정규화: 0.25, 0.09, 0.0225, 0.0025 / 0.365
    assert [round(x, 3) for x in t] == [0.685, 0.247, 0.062, 0.007]
