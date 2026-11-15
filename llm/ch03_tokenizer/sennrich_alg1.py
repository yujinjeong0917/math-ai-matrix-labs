"""대조용 기준 구현: Sennrich, Haddow, Birch(2016) Algorithm 1.

출처: "Neural Machine Translation of Rare Words with Subword Units", ACL 2016,
arXiv:1508.07909v5 (https://arxiv.org/abs/1508.07909v5), 확인일 2026-10-07.
원문 코드의 get_stats·merge_vocab을 그대로 두고, 바깥 루프만 함수로 감싸 print 대신
병합 목록을 돌려주게 했다. 쌍이 더 없으면 멈추는 줄 하나를 더했다.
공식 구현(subword-nmt)은 새 의존성이라 쓰지 않고, 논문에 실린 이 코드를 기준으로 삼는다.
"""

import re
from collections import defaultdict


def algorithm1_verbatim(vocab, num_merges):
    """Sennrich 등(2016) Algorithm 1을 원문 그대로 옮기고, print 대신 목록으로 돌려준다."""
    def get_stats(vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            # 단어를 "l o w </w>"처럼 공백으로 이은 문자열로 두고, 공백으로 기호를 나눈다.
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs

    def merge_vocab(pair, v_in):
        v_out = {}
        bigram = re.escape(" ".join(pair))
        # 쌍의 앞뒤가 공백이거나 문자열 경계일 때만 일치한다. 기호 일부(예: 'ab'의 b와 'bc')가 잘못 바뀌지 않게 한다.
        p = re.compile(r"(?<!\S)" + bigram + r"(?!\S)")
        for word in v_in:
            w_out = p.sub("".join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    out = []
    for _ in range(num_merges):
        pairs = get_stats(vocab)
        if not pairs:
            break
        # 동률이면 먼저 센 쌍(dict 삽입 순서). 이 구현의 learn_bpe_naive(tie='first')와 같은 규칙.
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        out.append(best)
    return out
