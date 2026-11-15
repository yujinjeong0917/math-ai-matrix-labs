"""4장 검증. `uv run pytest -q llm/ch04_embedding_mlp` 로 실행한다."""

import numpy as np
import torch

import ch04_grammar as G
import mlp_lm_np as M
import mlp_lm_torch as T
from ch04_ngram import InterpolatedTrigram

V = len(G.VOCAB)


def _setup(seed=0, B=12, m=3, h=5, direct=True):
    rng = np.random.default_rng(seed)
    p = M.init_params(V, m, G.CONTEXT, h, rng, direct)
    # p[k]에 작은 잡음을 더한다. direct=False일 때 W는 0으로 남겨 "직접 경로 없음"을 유지한다.
    for k in p:  # 편향까지 0이 아닌 값으로 채워 모든 경로를 검사한다
        p[k] = p[k] + 0.1 * rng.standard_normal(p[k].shape) if (direct or k != "W") else p[k]
    ctx = rng.integers(0, V, size=(B, G.CONTEXT))
    # 첫 예제는 두 문맥 자리에 같은 단어. C의 같은 행에 기울기가 두 번 더해지는 경로를 검사한다.
    ctx[0] = ctx[0, 0]  # 같은 단어가 두 자리에 나오는 경우(C 기울기가 더해지는 경로)도 넣는다
    tgt = rng.integers(0, V, size=B)
    return p, ctx, tgt


def test_hand_backprop_matches_central_difference():
    for direct in (True, False):
        p, ctx, tgt = _setup(direct=direct)
        _, cache = M.forward(p, ctx)
        g = M.backward(p, cache, tgt, direct)
        # 중앙 차분 (L(p+e) - L(p-e)) / 2e로 수치 미분을 하고 손 역전파와 비교한다. e = 1e-6.
        eps = 1e-6
        for name in p:
            if name == "W" and not direct:
                assert np.all(g["W"] == 0)
                continue
            num = np.zeros_like(p[name])
            it = np.nditer(p[name], flags=["multi_index"])
            for _ in it:
                i = it.multi_index
                old = p[name][i]
                p[name][i] = old + eps
                lp = M.loss(p, ctx, tgt)
                p[name][i] = old - eps
                lm = M.loss(p, ctx, tgt)
                p[name][i] = old
                num[i] = (lp - lm) / (2 * eps)
            # 상대 오차(노름 비). 분모가 0이 되지 않게 1e-12 바닥. 1e-6 미만이면 통과.
            rel = np.linalg.norm(num - g[name]) / max(np.linalg.norm(num) + np.linalg.norm(g[name]), 1e-12)
            assert rel < 1e-6, (name, rel)


def test_numpy_grads_match_torch_autograd():
    p, ctx, tgt = _setup(direct=True)
    _, cache = M.forward(p, ctx)
    g = M.backward(p, cache, tgt)
    model = T.NPLM.from_numpy(p)
    loss = T.loss(model, torch.from_numpy(ctx), torch.from_numpy(tgt))
    # 순전파 손실이 NumPy와 같아야 기울기 비교가 의미 있다.
    assert abs(loss.item() - M.loss(p, ctx, tgt)) < 1e-12
    loss.backward()
    gt = model.grads_as_numpy()
    for name in g:
        # 손 역전파와 PyTorch autograd 기울기의 최대 절대 차이.
        assert np.max(np.abs(g[name] - gt[name])) < 1e-10, name


def test_probabilities_sum_to_one_and_param_count():
    p, ctx, _ = _setup()
    P, _ = M.forward(p, ctx)
    np.testing.assert_allclose(P.sum(-1), 1.0, atol=1e-12)
    # 실제 파라미터 수와 원 논문 2절 공식이 일치하는지.
    assert sum(v.size for v in p.values()) == M.n_params(V, 3, G.CONTEXT, 5)


def test_unseen_pairs_never_in_training():
    S = G.make_splits(0)
    # 문장 인덱스: 0,1 = <s>, 2 = 때, 3 = 주어, 4 = 동사.
    seen = {(s[3], s[4]) for s in S["train"]}
    assert not seen & set(G.HELD_OUT)
    assert len(seen) == len(G.allowed_pairs())  # 빠진 쌍을 늘려도 맞도록


def test_ngram_gives_unseen_pair_only_unigram_mass():
    """학습에 없는 (강아지가, 잔다)는 트라이그램·바이그램 항이 0이고 l1 * P_add1(잔다)만 남는다."""
    S = G.make_splits(0)
    tr = G.to_examples(S["train"])
    ng = InterpolatedTrigram(V).fit(*tr)
    # (l3, l2, l1). 학습에 없는 쌍은 p3 = p2 = 0이라 l1 * p1만 남는다(기대값 expected).
    ng.lambdas = (0.05, 0.9, 0.05)
    I = G.IDX
    expected = 0.05 * (ng.c1[I["잔다"]] + 1) / (ng.N + V)
    assert abs(ng.prob(I["오늘"], I["강아지가"], I["잔다"]) - expected) < 1e-15
    total = sum(ng.prob(I["오늘"], I["강아지가"], w) for w in range(V))
    assert abs(total - 1.0) < 1e-12  # 보간해도 확률의 합은 1


def test_training_makes_cat_and_dog_closer_than_cat_and_student():
    S = G.make_splits(0)
    tr = G.to_examples(S["train"])
    rng = np.random.default_rng(0)
    p = M.init_params(V, 8, G.CONTEXT, 32, rng)
    # 3000스텝 학습 뒤 고양이와 강아지의 코사인 유사도가 고양이-학생보다 커야 한다(같은 동사와 쓰이므로). 시드 0에서의 확인.
    M.train(p, *tr, lr=0.3, steps=3000, batch=32, rng=rng)
    I = G.IDX
    C = p["C"]
    assert M.cosine(C[I["고양이가"]], C[I["강아지가"]]) > M.cosine(C[I["고양이가"]], C[I["학생이"]])
