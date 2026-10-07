"""6장 검증. `uv run pytest -q llm/ch06_attention_seq2seq` 로 실행한다."""

import numpy as np
import torch
import torch.nn.functional as F

import seq2seq_np as snp
import seq2seq_torch as stn


def _states(L=6, d=5, seed=0):
    rng = np.random.default_rng(seed)
    H = rng.standard_normal((L, d))
    s = rng.standard_normal(d)
    Ws, Wh, W = (rng.standard_normal((d, d)) / np.sqrt(d) for _ in range(3))
    v = rng.standard_normal(d) / np.sqrt(d)
    return s, H, {"additive": (Ws, Wh, v), "dot": (), "general": (W,)}


SCORES = {"additive": snp.additive_score, "dot": snp.dot_score, "general": snp.general_score}


def test_alpha_is_a_distribution_for_every_score():
    s, H, params = _states()
    for name, fn in SCORES.items():
        c, alpha = snp.attend(fn(s, H, *params[name]), H)
        assert np.all(alpha > 0)
        np.testing.assert_allclose(alpha.sum(), 1.0, atol=1e-12)
        np.testing.assert_allclose(c, alpha @ H, atol=1e-12)  # 문맥은 인코더 상태의 가중평균


def test_changing_one_encoder_state_only_moves_its_own_score():
    """h_k 하나만 바꾸면 점수도 e_k 하나만 바뀐다. 그래서 다른 위치끼리의 비율 alpha_i / alpha_j는 그대로다."""
    s, H, params = _states()
    k = 2
    for name, fn in SCORES.items():
        e1 = fn(s, H, *params[name])
        H2 = H.copy()
        H2[k] += 3.0 * s  # 디코더 상태 쪽으로 크게 민다
        e2 = fn(s, H2, *params[name])
        others = [i for i in range(len(H)) if i != k]
        np.testing.assert_allclose(e1[others], e2[others], atol=1e-12)
        a1, a2 = snp.softmax(e1), snp.softmax(e2)
        np.testing.assert_allclose(a1[others] / a1[others[0]], a2[others] / a2[others[0]], atol=1e-12)
    a1 = snp.softmax(snp.dot_score(s, H))
    H2 = H.copy()
    H2[k] += 3.0 * s
    assert snp.softmax(snp.dot_score(s, H2))[k] > a1[k]  # 곱셈형에서는 s 쪽으로 민 칸의 비율이 커진다


def test_numpy_scores_match_torch():
    s, H, params = _states()
    St, Ht = torch.tensor(s)[None, None], torch.tensor(H)[None]
    for name, fn in SCORES.items():
        ref = fn(s, H, *params[name])
        tfn = {"additive": stn.additive_score, "dot": stn.dot_score, "general": stn.general_score}[name]
        got = tfn(St, Ht, *[torch.tensor(p) for p in params[name]])[0, 0].numpy()
        np.testing.assert_allclose(got, ref, atol=1e-10)


def _np_weights(m):
    g = lambda mod, sfx: tuple(getattr(mod, f"{n}_l0{sfx}").detach().numpy() for n in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"))
    return g(m.enc, ""), g(m.dec, "")


def test_full_decoder_step_numpy_matches_torch():
    """같은 가중치로 인코딩 + 디코더 두 걸음을 돌려 logits와 alpha가 1e-10 안에서 같다(float64)."""
    torch.manual_seed(0)
    src = torch.randint(0, 20, (1, 7))
    for score in ("fixed", "additive", "dot", "general"):
        m = stn.Seq2Seq(20, d=8, e=6, score=score).double()
        gru_e, gru_d = _np_weights(m)
        E = m.emb.weight.detach().numpy()
        Wc, Wo = m.Wc.weight.detach().numpy(), m.Wo.weight.detach().numpy()
        params = tuple(p.detach().numpy() for p in m.score_params())

        H = snp.encode(E[src[0].numpy()], gru_e)
        Ht, st = m.encode_embedded(m.emb(src))
        np.testing.assert_allclose(Ht[0].detach().numpy(), H, atol=1e-10)

        s, y = H[-1], 20
        yt = torch.tensor([20])
        for _ in range(2):
            logits, s, alpha = snp.seq2seq_attention_step(E[y], s, H, gru_d, Wc, Wo, score=score, score_params=params)
            lt, st, at = m.step(yt, st, Ht)
            np.testing.assert_allclose(lt[0].detach().numpy(), logits, atol=1e-10)
            if alpha is not None:
                np.testing.assert_allclose(at[0].detach().numpy(), alpha, atol=1e-10)
            y = int(np.argmax(logits))
            yt = torch.tensor([y])


def test_teacher_forced_path_equals_step_loop():
    """학습에 쓰는 한 번의 nn.GRU 호출과, 평가에 쓰는 한 걸음씩 루프가 같은 logits를 낸다."""
    torch.manual_seed(1)
    src, tgt_in = torch.randint(0, 20, (3, 9)), torch.randint(0, 21, (3, 9))
    for score in ("fixed", "additive", "dot", "general"):
        m = stn.Seq2Seq(20, d=8, e=6, score=score).double()
        with torch.no_grad():
            full, _ = m(src, tgt_in)
            H, s = m.encode_embedded(m.emb(src))
            steps = []
            for t in range(tgt_in.shape[1]):
                lt, s, _ = m.step(tgt_in[:, t], s, H)
                steps.append(lt)
        np.testing.assert_allclose(torch.stack(steps, 1).numpy(), full.numpy(), atol=1e-10)


def test_score_flop_counts():
    assert snp.score_flops(10, 4, kind="dot") == 80
    assert snp.score_flops(10, 4, kind="general") == 32 + 80
    assert snp.score_flops(10, 4, kind="additive") == 32 + 40 + 80
    # 한 걸음의 점수 비용은 L에 비례한다. 출력도 L개면 문장 전체는 L^2에 비례한다.
    assert snp.score_flops(80, 64, kind="dot") == 2 * snp.score_flops(40, 64, kind="dot")


def _train_small(task, steps=500, L=8, seed=0):
    torch.manual_seed(seed)
    m = stn.Seq2Seq(20, d=32, e=16, score="dot")
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        x = torch.randint(0, 20, (64, L), generator=g)
        y = x if task == "copy" else x.flip(1)
        tgt_in = torch.cat([torch.full((64, 1), 20), y[:, :-1]], 1)
        logits, _ = m(x, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, 20), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
    x = torch.randint(0, 20, (200, L), generator=torch.Generator().manual_seed(99))
    y = x if task == "copy" else x.flip(1)
    pred, A = m.greedy(x, L)
    return float((pred == y).float().mean()), A


def test_trained_alignment_is_diagonal_for_copy_and_anti_diagonal_for_reverse():
    torch.set_num_threads(1)
    L = 8
    acc, A = _train_small("copy", L=L)
    assert acc > 0.9, acc
    assert float((A.argmax(-1) == torch.arange(L)).float().mean()) > 0.8  # 복사: 대각선

    acc, A = _train_small("reverse", L=L)
    assert acc > 0.9, acc
    offset = A.argmax(-1) - torch.arange(L).flip(0)
    # 역순: 반대 대각선 또는 그 한 칸 뒤. 단방향 인코더의 h_{j+1}에는 x_j가 아직 남아 있어서,
    # 모델이 정답 위치 j 대신 j+1을 가장 크게 봐도 답을 맞힌다(본문 6절의 관찰).
    assert float(((offset == 0) | (offset == 1)).float().mean()) > 0.9
