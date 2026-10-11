"""대화 템플릿 적용기와 손실 마스킹(가르치기용).

- render(): Qwen2.5-Instruct 대화 템플릿(도구 없는 분기)을 표준 라이브러리로 다시 쓴다. 템플릿의 글자들
  (<|im_start|>, <|im_end|>, 기본 system 문장)은 data/specs_2026-10-11.json 에서 읽는다.
- toy_tokens(): 가르치기용 토큰 세기. 특수 표시(<|im_start|>, <|im_end|>, <|endoftext|>)와 역할 이름
  (system, user, assistant)은 1개, 줄바꿈은 1개, 나머지는 글자(띄어쓰기 포함) 하나가 1개다.
  실제 Qwen 토크나이저의 수와는 다르다. 비율과 위치를 손으로 따라가려고 고른 규칙이다.
- labels(): 학습할 자리만 남기고 나머지를 -100(무시)으로 바꾼다. 남기는 자리는 TRL의
  qwen2_5_training.jinja 의 {% generation %} 블록과 같게 "'<|im_start|>assistant' 뒤의 줄바꿈 + 답 +
  <|im_end|> + 줄바꿈"이다. labels는 input과 길이가 같고, 한 칸 미는 일은 손실 계산 안에서 한다.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
SPEC_FILE = HERE / "data" / "specs_2026-10-11.json"
IGNORE = -100


def load_specs():
    return json.loads(SPEC_FILE.read_text(encoding="utf-8"))


_T = load_specs()["qwen2_5_instruct_template"]
IM_START, IM_END = _T["im_start"], _T["im_end"]
EOT = load_specs()["qwen2_5_base_tokenizer"]["eos_token"]   # <|endoftext|>
SPECIALS = (IM_START, IM_END, EOT)
ROLES = ("system", "user", "assistant")


def segments(messages, add_generation_prompt=False):
    """템플릿 결과를 (글자, 학습하나) 조각 목록으로 돌려준다. 이어 붙이면 render()와 같다."""
    segs = []
    msgs = list(messages)
    if not msgs or msgs[0]["role"] != "system":
        segs.append((IM_START + "system\n" + _T["default_system"] + IM_END + "\n", False))
    for m in msgs:
        if m["role"] == "assistant":
            segs.append((IM_START + "assistant", False))
            segs.append(("\n" + m["content"] + IM_END + "\n", True))
        else:
            segs.append((IM_START + m["role"] + "\n" + m["content"] + IM_END + "\n", False))
    if add_generation_prompt:
        segs.append((_T["generation_prompt"], False))
    return segs


def render(messages, add_generation_prompt=False):
    return "".join(s for s, _ in segments(messages, add_generation_prompt))


def toy_tokens(text):
    """가르치기용 토큰 목록."""
    out, i = [], 0
    while i < len(text):
        for sp in SPECIALS:
            if text.startswith(sp, i):
                out.append(sp)
                i += len(sp)
                # 특수 표시 바로 뒤의 역할 이름은 1개로 센다
                for r in ROLES:
                    if sp == IM_START and text.startswith(r, i):
                        out.append(r)
                        i += len(r)
                        break
                break
        else:
            out.append(text[i])
            i += 1
    return out


def tokens_and_mask(messages):
    """(토큰 목록, 학습하나 목록). 조각마다 따로 세서 조각 경계가 토큰 경계와 같게 한다."""
    toks, mask = [], []
    for s, train in segments(messages):
        t = toy_tokens(s)
        toks += t
        mask += [train] * len(t)
    return toks, mask


def labels(token_ids, mask):
    return [tid if m else IGNORE for tid, m in zip(token_ids, mask)]


def counts(messages):
    toks, mask = tokens_and_mask(messages)
    return {"total": len(toks), "trained": sum(mask), "ignored": len(toks) - sum(mask)}


def masked_mean(per_token_loss, mask):
    """학습하는 자리만 평균. PyTorch CrossEntropyLoss(ignore_index=-100, reduction='mean')와 같은 셈."""
    kept = [l for l, m in zip(per_token_loss, mask) if m]
    return sum(kept) / len(kept)


# ---------- 패딩과 끝 표시 ----------

def pad_batch(token_lists, pad_tok):
    n = max(len(t) for t in token_lists)
    return [t + [pad_tok] * (n - len(t)) for t in token_lists]


def labels_mask_by_pad_value(padded, pad_tok):
    """흔한 손짜기 실수: '패드와 같은 값이면 -100'으로 가린다. 패드가 끝 표시와 같은 토큰이면 끝 표시도 가려진다."""
    return [[IGNORE if t == pad_tok else t for t in row] for row in padded]


def labels_mask_by_length(padded, lengths):
    """바른 방법: 원래 길이 뒤의 자리만 가린다(attention mask와 같은 정보)."""
    return [[t if i < n else IGNORE for i, t in enumerate(row)] for row, n in zip(padded, lengths)]
