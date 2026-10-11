"""내 데이터로 LLM 파인튜닝하기 9장: 파인튜닝 결과 평가의 계산 모형.

표준 라이브러리만 쓴다. 무작위는 random.Random(seed)로만 쓰니 결과가 매번 같다.

- 손실 곡선: 식으로 만든 가상 학습·검증 손실, transformers EarlyStoppingCallback(v5.19.0)과 같은 규칙의 조기 종료
- 채점: 답변 조립, 사람 채점표(인사·맺음말·길이·사실·규칙·말투), 정확 일치, 정규화 일치, 규칙 검사
- 비교: 실패 종류별 바뀜 표, 순서를 바꾼 장난감 채점자, 짝지은 부트스트랩, 엇갈린 문항의 부호 검정
"""

import json
import math
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
SPEC_FILE = DATA / "specs_2026-10-11.json"
INQUIRIES = HERE.parent / "ch01_when_finetune" / "data" / "inquiries.json"
TYPES = ["K", "F", "T", "I", "R", "OK"]


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def load_specs():
    return json.loads(SPEC_FILE.read_text(encoding="utf-8"))


def load_inquiries():
    """1장 평가 세트(문의 40건). 복사하지 않고 그대로 읽는다."""
    return json.loads(INQUIRIES.read_text(encoding="utf-8"))["items"]


# ---------- 손실 곡선 ----------

def train_loss(s, p, noise=0.0):
    return p["floor"] + p["amp"] * math.exp(-s / p["tau"]) + noise


def val_loss(s, p, noise=0.0):
    return (p["floor"] + p["amp"] * math.exp(-s / p["tau"])
            + p["rise_coef"] * max(0, s - p["rise_start"]) ** 2 + noise)


def make_curves(cfg, steps_per_epoch):
    """학습 손실은 스텝마다(logging_steps=1), 검증 손실은 eval_every 스텝마다 만든다."""
    total = cfg["epochs"] * steps_per_epoch
    rt = random.Random(cfg["train"]["seed"])
    rv = random.Random(cfg["val"]["seed"])
    train = [round(train_loss(s, cfg["train"], rt.gauss(0, cfg["train"]["noise_sd"])), 4)
             for s in range(1, total + 1)]
    evals = []
    for s in range(cfg["eval_every"], total + 1, cfg["eval_every"]):
        evals.append({"step": s,
                      "val": round(val_loss(s, cfg["val"], rv.gauss(0, cfg["val"]["noise_sd"])), 4),
                      "train_smooth": round(train_loss(s, cfg["train"]), 4),
                      "leaky_val": round(train_loss(s, cfg["train"]) + cfg["leaky_val_offset"], 4)})
    return {"total_steps": total, "train": train, "evals": evals}


def early_stopping_trace(values, patience=1, threshold=0.0, greater_is_better=False):
    """transformers v5.19.0의 EarlyStoppingCallback.check_metric_value와 Trainer 순서를 그대로 옮겼다.

    평가할 때마다 (1) 콜백이 '이전까지의 가장 좋은 값(best)'과 비교해 카운터를 고치고,
    (2) 그다음 Trainer가 threshold 없이 조금이라도 좋아졌으면 best를 바꾼다.
    카운터가 patience 이상이 되면 그 평가에서 멈춘다. 반환: 평가마다의 기록, 멈춘 위치(없으면 None), best 위치.
    """
    better = (lambda a, b: a > b) if greater_is_better else (lambda a, b: a < b)
    best = None
    best_i = None
    counter = 0
    rows = []
    stop = None
    for i, v in enumerate(values):
        prev = best
        if best is None or (better(v, best) and abs(v - best) > threshold):
            counter = 0
        else:
            counter += 1
        if best is None or better(v, best):  # _determine_best_metric
            best, best_i = v, i
        rows.append({"i": i, "value": v, "prev_best": prev, "counter": counter, "best_after": best})
        if counter >= patience:
            stop = i
            break
    return {"rows": rows, "stop": stop, "best": best_i}


def first_below(values, limit):
    for i, v in enumerate(values):
        if v < limit:
            return i
    return None


# ---------- 답변 조립 ----------

def punct_variant(body):
    """띄어쓰기 하나와 마지막 문장부호만 바꾼다. 뜻은 그대로다."""
    out = body.replace(". ", ".  ", 1)
    if out.endswith("."):
        out = out[:-1] + "!"
    return out


def body_text(bank, qid, variant):
    item = bank["items"][qid]
    if variant == "ref_punct":
        return punct_variant(item["ref"])
    if variant == "offtopic":
        return bank["offtopic"]
    return item[variant]


def compose(bank, qid, variant, greet=True, sign=True):
    parts = []
    if greet:
        parts.append(bank["greeting"])
    parts.append(body_text(bank, qid, variant))
    if sign:
        parts.append(bank["signoff"])
    return "\n".join(parts)


def reference(bank, qid):
    return compose(bank, qid, "ref", True, True)


def sentences(text):
    return [s for s in re.split(r"(?<=[.?!])\s+", text.strip()) if s]


# ---------- 사람 채점표 ----------

RUBRIC = ["인사", "맺음말", "길이", "사실·계산", "규칙", "말투", "질문에 맞음"]


def human_rubric(bank, qid, variant, greet, sign):
    """가상 사람 채점. 본문 종류가 무엇을 어겼는지로 칸을 정한다(사람이 읽고 판단했다고 두는 값)."""
    r = {
        "인사": greet,
        "맺음말": sign,
        "길이": len(sentences(body_text(bank, qid, variant))) <= bank["max_body_sentences"],
        "사실·계산": variant != "wrong",
        "규칙": variant != "forbidden",
        "말투": variant != "curt",
        "질문에 맞음": variant != "offtopic",
    }
    r["통과"] = all(r[k] for k in RUBRIC)
    return r


# ---------- 자동 채점 ----------

_NORM = re.compile(r"[\s.,!?~·]")


def exact_match(answer, ref):
    return answer == ref


def normalized_match(answer, ref):
    return _NORM.sub("", answer) == _NORM.sub("", ref)


def rule_check(bank, qid, answer):
    """규칙 검사: 인사로 시작, 맺음말로 끝, 본문 3문장 이하, must 글자 모두 있음, forbid 글자 없음.
    말투와 '질문에 맞는 답인지'는 보지 못한다."""
    item = bank["items"][qid]
    g, s = bank["greeting"], bank["signoff"]
    starts, ends = answer.startswith(g), answer.endswith(s)
    body = answer
    if starts:
        body = body[len(g):]
    if ends:
        body = body[: len(body) - len(s)]
    body = body.strip()
    r = {
        "인사": starts,
        "맺음말": ends,
        "길이": len(sentences(body)) <= bank["max_body_sentences"],
        "필수 글자": all(m in body for m in item.get("must", [])),
        "금지 글자 없음": not any(f in body for f in item.get("forbid", [])),
    }
    r["통과"] = all(r.values())
    return r


def grade_model(bank, sheet_model, inquiries):
    rows = []
    for it in inquiries:
        qid = it["id"]
        variant, greet, sign = sheet_model[qid]
        ans = compose(bank, qid, variant, greet, sign)
        ref = reference(bank, qid)
        h = human_rubric(bank, qid, variant, greet, sign)
        rc = rule_check(bank, qid, ans)
        rows.append({"id": qid, "type": it["failure"], "variant": variant, "answer": ans,
                     "exact": exact_match(ans, ref), "normalized": normalized_match(ans, ref),
                     "rule": rc["통과"], "rule_detail": rc, "human": h["통과"], "human_detail": h})
    return rows


def summarize(rows):
    n = len(rows)
    return {"n": n,
            "exact": sum(r["exact"] for r in rows),
            "normalized": sum(r["normalized"] for r in rows),
            "rule": sum(r["rule"] for r in rows),
            "human": sum(r["human"] for r in rows),
            "rule_false_pass": [r["id"] for r in rows if r["rule"] and not r["human"]],
            "rule_false_fail": [r["id"] for r in rows if not r["rule"] and r["human"]],
            "rule_agree": sum(r["rule"] == r["human"] for r in rows),
            **rule_vs_human(rows)}


def rule_vs_human(rows):
    """사람 판정을 정답으로 두고 규칙 검사의 혼동행렬. '통과'를 양성으로 본다."""
    tp = sum(r["rule"] and r["human"] for r in rows)
    fp = sum(r["rule"] and not r["human"] for r in rows)
    fn = sum((not r["rule"]) and r["human"] for r in rows)
    tn = sum((not r["rule"]) and (not r["human"]) for r in rows)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "recall": round(tp / (tp + fn), 4) if tp + fn else None}


def by_type(rows_a, rows_b):
    """실패 종류별로 a(기준)와 b(새 모델)의 통과 수, 틀림→맞음, 맞음→틀림."""
    out = {}
    for t in TYPES:
        pa = [r for r in rows_a if r["type"] == t]
        pb = {r["id"]: r for r in rows_b if r["type"] == t}
        out[t] = {"n": len(pa),
                  "a_pass": sum(r["human"] for r in pa),
                  "b_pass": sum(pb[r["id"]]["human"] for r in pa),
                  "fixed": [r["id"] for r in pa if not r["human"] and pb[r["id"]]["human"]],
                  "broke": [r["id"] for r in pa if r["human"] and not pb[r["id"]]["human"]]}
    return out


# ---------- 일반 문항(잊어버림) ----------

def probe_pass(answer, expect):
    a = answer.lower()
    return any(e.lower() in a for e in expect)


# ---------- 장난감 채점자(순서 치우침) ----------

def toy_judge(human_x, human_y, x_first):
    """사람 판정이 갈리면 사람이 고른 쪽을, 둘 다 통과이거나 둘 다 탈락이면 먼저 보여 준 쪽을 고른다.
    Zheng 외(2023)가 보고한 '순서 치우침'만 흉내 낸 장난감이다([가정]). 반환: x를 골랐으면 True."""
    if human_x != human_y:
        return human_x
    return x_first


def judge_both_orders(hx, hy):
    xf = [toy_judge(a, b, True) for a, b in zip(hx, hy)]
    yf = [toy_judge(a, b, False) for a, b in zip(hx, hy)]
    n = len(hx)
    consistent = [a == b for a, b in zip(xf, yf)]
    return {"n": n, "x_wins_x_first": sum(xf), "x_wins_y_first": sum(yf),
            "x_wins_avg": (sum(xf) + sum(yf)) / 2, "inconsistent": n - sum(consistent),
            "x_wins_consistent": sum(a and c for a, c in zip(xf, consistent)),
            "y_wins_consistent": sum((not a) and c for a, c in zip(xf, consistent))}


# ---------- 생성 설정 장난감 ----------

def sampled_pass_counts(greedy_pass, p_hit, p_miss, runs, seed0):
    """무작위로 뽑아 쓰면(do_sample=True) 문항마다 통과가 확률이 된다고 둔 장난감([가정])."""
    out = []
    for k in range(runs):
        rng = random.Random(seed0 + k)
        out.append(sum(rng.random() < (p_hit if g else p_miss) for g in greedy_pass))
    return out


# ---------- 짝지은 부트스트랩과 부호 검정 ----------

def percentile(sorted_vals, q):
    """NumPy 기본(linear)과 같은 방식의 백분위수. q는 0~100."""
    n = len(sorted_vals)
    pos = (n - 1) * q / 100
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


def paired_bootstrap(a, b, n_boot=1000, seed=0, level=0.95):
    """a, b는 같은 문항 순서의 0/1 통과 목록. 문항 위치를 중복을 허락해 n개 다시 뽑고, 두 모델에 같은 위치를 쓴다.
    반환: 통과율 차이 b - a, 다시 뽑은 차이의 가운데 level 구간."""
    n = len(a)
    assert n == len(b)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    tail = (1 - level) / 2 * 100
    lo, hi = percentile(diffs, tail), percentile(diffs, 100 - tail)
    return {"n": n, "diff": (sum(b) - sum(a)) / n, "lo": lo, "hi": hi,
            "contains_zero": lo <= 0 <= hi, "n_boot": n_boot, "seed": seed}


def discordant(a, b):
    """(a만 통과, b만 통과) 문항 수."""
    return sum(1 for x, y in zip(a, b) if x and not y), sum(1 for x, y in zip(a, b) if y and not x)


def sign_test(a_only, b_only):
    """엇갈린 문항만 보고, 둘이 같은 실력이면 각 문항이 동전 던지기라고 둔 정확한 양측 검정(McNemar 정확 검정과 같다)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def copies_needed(a_only, b_only, alpha=0.05, max_k=50):
    """엇갈린 비율이 그대로일 때, 평가 세트를 몇 배로 늘려야 부호 검정이 alpha 아래로 내려가나."""
    for k in range(1, max_k + 1):
        if sign_test(a_only * k, b_only * k) < alpha:
            return k
    return None
