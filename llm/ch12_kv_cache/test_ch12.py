"""12장 검증. `uv run pytest -q llm/ch12_kv_cache` 로 실행한다."""

import numpy as np
import torch

import kvcache_np as kv
import kvcache_torch as kt


def _params(h=4, h_kv=None, seed=0, n_ctx=64):
    return kv.init_params(vocab=20, n_ctx=n_ctx, d=16, L=3, h=h, h_kv=h_kv, seed=seed)


def test_cached_logits_match_no_cache_every_step():
    """캐시 있음/없음: 모든 스텝의 로짓이 1e-10 안에서 같다(float64). MHA, GQA, MQA 모두."""
    for h_kv in (4, 2, 1):
        p = _params(h_kv=h_kv)
        seq_a, la = kv.generate_no_cache(p, [3, 1, 4], 40, record_logits=True)
        seq_b, lb = kv.generate_with_cache(p, [3, 1, 4], 40, record_logits=True)
        assert seq_a == seq_b
        assert max(np.abs(x - y).max() for x, y in zip(la, lb)) < 1e-10


def test_same_seed_sampling_gives_same_tokens():
    p = _params()
    a = kv.generate_no_cache(p, [0], 50, rng=np.random.default_rng(7))
    b = kv.generate_with_cache(p, [0], 50, rng=np.random.default_rng(7))
    assert a == b


def test_prefix_kv_unchanged_by_later_tokens_in_every_layer():
    """causal 마스크 아래에서는 뒤에 토큰이 붙어도 앞 위치의 k, v가 모든 층에서 그대로다.
    1층은 입력 임베딩만 보므로 당연하고, 확인할 가치가 있는 건 2층 이상이다."""
    p = _params()
    rng = np.random.default_rng(1)
    toks = rng.integers(0, 20, 30)
    _, short = kv.forward(p, toks[:10], return_kv=True)
    _, long = kv.forward(p, toks, return_kv=True)
    for l in range(1, 3):
        for a, b in zip(short[l], long[l]):
            np.testing.assert_allclose(b[:10], a, atol=1e-12)


def test_without_causal_mask_deeper_kv_would_change():
    """대조: 마스크를 빼면 2층부터는 앞 위치의 k, v가 뒤 토큰에 따라 바뀐다. 그래서 캐시는 마스크에 기댄다."""
    p = _params()
    toks = np.random.default_rng(1).integers(0, 20, 30)
    _, short = kv.forward(p, toks[:10], causal=False, return_kv=True)
    _, long = kv.forward(p, toks, causal=False, return_kv=True)
    np.testing.assert_allclose(long[0][0][:10], short[0][0], atol=1e-12)  # 1층은 여전히 같다
    assert np.abs(long[1][0][:10] - short[1][0]).max() > 1e-3


def test_mqa_with_one_head_equals_mha():
    """헤드가 1개면 키·값을 공유할 상대가 없으니 MQA와 MHA가 같은 모델이다."""
    a, b = _params(h=1, h_kv=1, seed=3), _params(h=1, seed=3)
    toks = [1, 2, 3, 4, 5, 6]
    np.testing.assert_allclose(kv.forward(a, toks), kv.forward(b, toks), atol=1e-12)


def test_mqa_equals_mha_with_tied_kv_heads():
    """MQA는 '모든 헤드가 같은 Wk, Wv를 쓰는 MHA'와 같다."""
    mqa = _params(h=4, h_kv=1, seed=5)
    mha = _params(h=4, seed=5)
    for pm, pq in zip(mha["layers"], mqa["layers"]):
        for k in pm:
            if k in ("Wk", "Wv"):
                pm[k] = np.tile(pq[k], (1, 4))  # 헤드 4개에 같은 열을 복제
            else:
                pm[k] = pq[k]
    for k in ("tok", "pos", "gf", "bf", "Wout", "bout"):
        mha[k] = mqa[k]
    toks = [2, 7, 1, 8, 2, 8]
    np.testing.assert_allclose(kv.forward(mha, toks), kv.forward(mqa, toks), atol=1e-12)


def test_cache_bytes_matches_array_nbytes():
    for h_kv in (4, 1):
        p = _params(h_kv=h_kv)
        cache = kv.new_cache(p, 64)
        kv.generate_with_cache(p, [1, 2], 30, cache=cache)
        c = p["cfg"]
        filled = 2 + 30 - 1  # 마지막으로 뽑은 토큰은 아직 순전파하지 않았다
        assert cache.nbytes() == kv.cache_bytes(c["L"], filled, h_kv, c["dk"], 8)


def test_positions_and_flops_counts():
    assert kv.positions_processed(1, 4, cached=False) == 1 + 2 + 3 + 4
    assert kv.positions_processed(1, 4, cached=True) == 4
    args = dict(d=16, L=3, h=4, h_kv=4, dk=4, vocab=20)
    assert kv.flops_no_cache(1, 1, **args) == kv.flops_with_cache(1, 1, **args)  # 한 스텝이면 같다
    assert kv.flops_no_cache(1, 50, **args) > kv.flops_with_cache(1, 50, **args)


def test_numpy_matches_torch_and_torch_caches_agree():
    for h_kv in (4, 1):
        p = _params(h_kv=h_kv)
        m = kt.from_numpy(p)
        toks = [5, 3, 9, 1, 0, 2, 7]
        ref = kv.forward(p, toks)
        out = m(torch.tensor([toks])).detach().numpy()[0]
        np.testing.assert_allclose(out, ref, atol=1e-10)
        prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
        seqs = [kt.generate(m, prompt, 25, mode) for mode in ("none", "cat", "prealloc")]
        assert torch.equal(seqs[0], seqs[1]) and torch.equal(seqs[0], seqs[2])
        np_seq = kv.generate_with_cache(p, [1, 2, 3], 25)
        assert seqs[2][0].tolist() == np_seq
