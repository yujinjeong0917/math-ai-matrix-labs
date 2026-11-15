"""12장 검증. `uv run pytest -q llm/ch12_kv_cache` 로 실행한다."""

import numpy as np
import torch

import kvcache_np as kv
import kvcache_torch as kt


def _params(h=4, h_kv=None, seed=0, n_ctx=64):
    # 작은 설정: 어휘 20, d 16, 3층. n_ctx 64는 프롬프트 3 + 생성 40이 들어가기에 충분
    return kv.init_params(vocab=20, n_ctx=n_ctx, d=16, L=3, h=h, h_kv=h_kv, seed=seed)


def test_cached_logits_match_no_cache_every_step():
    """캐시 있음/없음: 모든 스텝의 로짓이 1e-10 안에서 같다(float64). MHA, GQA, MQA 모두."""
    for h_kv in (4, 2, 1):
        p = _params(h_kv=h_kv)
        # record_logits=True: 스텝마다 마지막 위치 로짓을 저장해 두 방식을 스텝 단위로 대조
        seq_a, la = kv.generate_no_cache(p, [3, 1, 4], 40, record_logits=True)
        seq_b, lb = kv.generate_with_cache(p, [3, 1, 4], 40, record_logits=True)
        # 탐욕이라 토큰 열이 정확히 같아야 한다
        assert seq_a == seq_b
        # float64이므로 1e-10 이내 일치(행렬곱 순서 차이로 비트 단위 일치까지는 요구하지 않음)
        assert max(np.abs(x - y).max() for x, y in zip(la, lb)) < 1e-10


def test_same_seed_sampling_gives_same_tokens():
    p = _params()
    # 같은 시드의 표집: 로짓이 거의 같으면 같은 토큰이 뽑힌다. 반올림 오차로 경계값이 갈릴 위험은 이 시드에서는 나타나지 않았다고 가정(확인 필요)
    a = kv.generate_no_cache(p, [0], 50, rng=np.random.default_rng(7))
    b = kv.generate_with_cache(p, [0], 50, rng=np.random.default_rng(7))
    assert a == b


def test_prefix_kv_unchanged_by_later_tokens_in_every_layer():
    """causal 마스크 아래에서는 뒤에 토큰이 붙어도 앞 위치의 k, v가 모든 층에서 그대로다.
    1층은 입력 임베딩만 보므로 당연하고, 확인할 가치가 있는 건 2층 이상이다."""
    p = _params()
    rng = np.random.default_rng(1)
    toks = rng.integers(0, 20, 30)
    # 토큰 10개짜리 접두사와 30개 전체에서 층별 k, v를 비교한다
    _, short = kv.forward(p, toks[:10], return_kv=True)
    _, long = kv.forward(p, toks, return_kv=True)
    # 층 인덱스 1, 2(2층, 3층). 0(1층)은 임베딩만 보므로 제외
    for l in range(1, 3):
        for a, b in zip(short[l], long[l]):
            # 긴 입력의 앞 10위치 k, v == 짧은 입력의 것. causal이라 뒤 토큰이 앞 위치에 영향을 주지 않는다
            np.testing.assert_allclose(b[:10], a, atol=1e-12)


def test_without_causal_mask_deeper_kv_would_change():
    """대조: 마스크를 빼면 2층부터는 앞 위치의 k, v가 뒤 토큰에 따라 바뀐다. 그래서 캐시는 마스크에 기댄다."""
    p = _params()
    toks = np.random.default_rng(1).integers(0, 20, 30)
    # causal=False: 모든 위치가 서로 보는 양방향 어텐션
    _, short = kv.forward(p, toks[:10], causal=False, return_kv=True)
    _, long = kv.forward(p, toks, causal=False, return_kv=True)
    # 1층의 k는 임베딩에서만 나오므로 마스크가 없어도 같다
    np.testing.assert_allclose(long[0][0][:10], short[0][0], atol=1e-12)  # 1층은 여전히 같다
    # 2층부터는 앞 위치가 뒤 토큰을 보게 되어 k가 달라진다(1e-3은 임의의 '눈에 띄는 차이' 기준)
    assert np.abs(long[1][0][:10] - short[1][0]).max() > 1e-3


def test_mqa_with_one_head_equals_mha():
    """헤드가 1개면 키·값을 공유할 상대가 없으니 MQA와 MHA가 같은 모델이다."""
    # 같은 시드라 같은 난수 순서로 가중치가 만들어진다(h=1이면 두 설정의 모양이 같다)
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
                # Wk (d, dk)를 열 방향으로 4번 복제 -> (d, 4*dk): 헤드 4개가 같은 Wk를 갖는다
                pm[k] = np.tile(pq[k], (1, 4))  # 헤드 4개에 같은 열을 복제
            else:
                pm[k] = pq[k]
    # 가중치를 완전히 같게 하려고 나머지도 복사
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
        # 마지막으로 뽑은 토큰은 순전파되지 않아 캐시에 없다: 프롬프트 2 + 생성 30 - 1 = 31칸
        filled = 2 + 30 - 1  # 마지막으로 뽑은 토큰은 아직 순전파하지 않았다
        # 실제 배열 바이트 수 == 공식 2*L*n*h_kv*dk*8
        assert cache.nbytes() == kv.cache_bytes(c["L"], filled, h_kv, c["dk"], 8)


def test_positions_and_flops_counts():
    # 캐시 없음: 접두사 길이 1, 2, 3, 4를 모두 다시 처리 = 10
    assert kv.positions_processed(1, 4, cached=False) == 1 + 2 + 3 + 4
    # 캐시 있음: 프리필 1 + 디코드 3 = 4
    assert kv.positions_processed(1, 4, cached=True) == 4
    args = dict(d=16, L=3, h=4, h_kv=4, dk=4, vocab=20)
    # 생성 1토큰이면 프리필만 있어 두 방식이 같다
    assert kv.flops_no_cache(1, 1, **args) == kv.flops_with_cache(1, 1, **args)  # 한 스텝이면 같다
    assert kv.flops_no_cache(1, 50, **args) > kv.flops_with_cache(1, 50, **args)


def test_numpy_matches_torch_and_torch_caches_agree():
    for h_kv in (4, 1):
        p = _params(h_kv=h_kv)
        # 같은 가중치의 float64 torch 모델
        m = kt.from_numpy(p)
        toks = [5, 3, 9, 1, 0, 2, 7]
        ref = kv.forward(p, toks)
        out = m(torch.tensor([toks])).detach().numpy()[0]
        np.testing.assert_allclose(out, ref, atol=1e-10)
        prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
        # 세 방식(캐시 없음, cat, 미리 할당)이 같은 토큰 열을 내는지. 배치 2개 프롬프트
        seqs = [kt.generate(m, prompt, 25, mode) for mode in ("none", "cat", "prealloc")]
        assert torch.equal(seqs[0], seqs[1]) and torch.equal(seqs[0], seqs[2])
        np_seq = kv.generate_with_cache(p, [1, 2, 3], 25)
        # torch 첫 배치 행이 NumPy 생성 결과와 같다
        assert seqs[2][0].tolist() == np_seq
