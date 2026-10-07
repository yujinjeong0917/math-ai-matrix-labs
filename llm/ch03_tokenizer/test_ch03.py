"""3장 검증. `uv run pytest -q llm/ch03_tokenizer` 로 실행한다."""

import numpy as np

import bpe_np as B
import ch03_data as D
from ch03_ngram import InterpTrigram
from sennrich_alg1 import algorithm1_verbatim


def _small_corpus(lang="en_like"):
    sents = D.make_corpus(lang, n_sentences=600, n_stems=300, seed=3)
    return D.split(sents, 0)


def test_figure1_lower_is_low_er():
    """Sennrich 등 Figure 1: 사전 {low, lowest, newer, wider}에서 배운 병합으로 'lower' → 'low er·'."""
    fig1 = {"low": 1, "lowest": 1, "newer": 1, "wider": 1}
    tok = B.BPETokenizer(fig1, B.learn_bpe(fig1, 4))
    assert tok.apply_bpe("lower") == ["low", "er" + B.EOW]


def test_matches_paper_algorithm1_when_tie_rule_matches():
    """논문 Algorithm 1 원문 코드와, 동률 규칙만 맞춘 우리 구현의 병합 목록이 같다."""
    toy = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    alg1 = algorithm1_verbatim({"l o w </w>": 5, "l o w e r </w>": 2, "n e w e s t </w>": 6, "w i d e s t </w>": 3}, 10)
    assert B.learn_bpe_naive(toy, 10, tie="first") == alg1


def test_fast_equals_naive_and_is_deterministic():
    tr, _, _ = _small_corpus()
    f = B.word_freqs(tr)
    naive = B.learn_bpe_naive(f, 120)
    assert B.learn_bpe(f, 120) == naive
    assert B.learn_bpe(f, 120) == B.learn_bpe(f, 120)


def test_vocab_size_is_alphabet_plus_merges():
    tr, _, _ = _small_corpus()
    f = B.word_freqs(tr)
    for M in (0, 50, 200):
        tk = B.BPETokenizer(f, B.learn_bpe(f, M))
        n_new = len({a + b for a, b in tk.merges} - set(tk.alphabet) - {B.EOW})
        assert tk.vocab_size == len(tk.alphabet) + 1 + n_new + 1  # 글자 + </w> + 병합 + <unk>
        assert n_new == M  # 이 말뭉치에서는 병합마다 새 기호가 하나씩 생긴다


def test_roundtrip_and_no_unk_on_seen_alphabet():
    for lang in ("en_like", "ko_like"):
        tr, _, te = _small_corpus(lang)
        f = B.word_freqs(tr)
        tk = B.BPETokenizer(f, B.learn_bpe(f, 300))
        for s in tr[:50]:
            assert tk.decode(tk.encode(s)) == s
        if lang == "en_like":  # 알파벳 26자가 모두 학습에 있으므로 테스트에도 <unk>가 없다
            assert all(B.UNK not in tk.tokenize(s) for s in te)


def test_unseen_character_becomes_unk():
    f = {"가나": 3, "나다": 2}
    tk = B.BPETokenizer(f, B.learn_bpe(f, 2))
    assert tk.tokenize("가힣") == ["가", B.UNK, B.EOW]


def test_word_vocab_unk_rate_shrinks_with_V():
    tr, _, te = _small_corpus()
    f = B.word_freqs(tr)
    rates = [B.unk_rate(te, B.word_vocab(f, V)) for V in (50, 200, 1000)]
    assert rates[0] > rates[1] > rates[2] >= 0


def test_interpolated_trigram_is_a_distribution():
    tr, _, _ = _small_corpus()
    seqs = [s.split(" ") for s in tr[:200]]
    vocab = sorted({w for s in seqs for w in s})
    lm = InterpTrigram(seqs, vocab)
    for u, v in [("<s>", "<s>"), ("<s>", seqs[0][0]), (seqs[0][0], seqs[0][1]), ("없는말", "없는말")]:
        total = sum(lm.prob(u, v, w, (0.99, 0.5, 0.7)) for w in lm.vocab)
        assert abs(total - 1) < 1e-9
    comps = lm.components(seqs[:5])
    direct = [lm.prob(*(["<s>", "<s>"] + s + ["</s>"])[i - 2:i + 1], (0.99, 0.5, 0.7))
              for s in seqs[:5] for i in range(2, len(s) + 3)]
    np.testing.assert_allclose(lm.probs(comps, (0.99, 0.5, 0.7)), direct, rtol=1e-12)


def test_op_screen_hand_example():
    """웹 챕터 첫 화면의 손 계산(단어 끝 표시 없이): 학교에 4, 학교는 3, 학생은 2."""
    words = {("학", "교", "에"): 4, ("학", "교", "는"): 3, ("학", "생", "은"): 2}

    def counts(ws):
        c = {}
        for w, f in ws.items():
            for p in zip(w[:-1], w[1:]):
                c[p] = c.get(p, 0) + f
        return c

    c1 = counts(words)
    assert c1 == {("학", "교"): 7, ("교", "에"): 4, ("교", "는"): 3, ("학", "생"): 2, ("생", "은"): 2}
    words = {tuple(B._merge_word(list(w), ("학", "교"))): f for w, f in words.items()}
    c2 = counts(words)
    assert max(c2.values()) == 4 and c2[("학교", "에")] == 4 and list(c2.values()).count(4) == 1
    alphabet = {"학", "교", "에", "는", "생", "은"}
    assert len(alphabet) + 2 == 8
    # 새 단어 '학생에': 두 병합 모두 해당 없음 → 세 조각
    w = ["학", "생", "에"]
    for m in (("학", "교"), ("학교", "에")):
        w = B._merge_word(w, m)
    assert w == ["학", "생", "에"]
