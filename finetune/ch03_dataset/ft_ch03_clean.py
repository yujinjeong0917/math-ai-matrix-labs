"""정제, 중복·거의 중복 찾기, 평가 세트 누수 검사, 학습·검증 나누기.

닮음 재기: 1장 검색기와 같은 단위(어절마다 이웃한 두 글자, 한 글자 어절은 그대로)의 집합을 만들고
자카드 유사도 = |A ∩ B| ÷ |A ∪ B| 로 잰다. 집합은 늘 정렬해서 다루고, 무작위는 random.Random(seed)와
hashlib 만 쓴다(파이썬 hash()는 실행마다 바뀌어서 쓰지 않는다).
"""

import hashlib
import json
import random
import re
from pathlib import Path

HERE = Path(__file__).parent
RAW_FILE = HERE / "data" / "raw_pairs.json"
EVAL_FILE = HERE.parent / "ch01_when_finetune" / "data" / "inquiries.json"

_ORDER = re.compile(r"\b[A-Z]\d{4}\b")
_PUNCT = re.compile(r"[^\w\s]")
_JAMO = re.compile(r"[ㄱ-ㆎ]+")          # ㅠㅠ, ㅋㅋ 같은 낱자
_SPACE = re.compile(r"\s+")
# 1장에서 "자주 바뀌니 검색에 맡기자"고 한 정책 값(환불 기한, 배송비, 교환비, 적립률, 쿠폰 기한)
VOLATILE_NUM = re.compile(r"\d+영업일|\d[\d,]*원|\d+%|\d+일 안에|\d+회")
VOLATILE_KEY = re.compile(r"환불|돌려|배송비|교환|적립금|쿠폰|무료")
PLACEHOLDERS = {"", "확인 중", "확인중", "-"}


def load_raw():
    return json.loads(RAW_FILE.read_text(encoding="utf-8"))["items"]


def load_eval():
    return json.loads(EVAL_FILE.read_text(encoding="utf-8"))["items"]


def normalize(text):
    t = _ORDER.sub("주문번호값", text)
    t = _JAMO.sub(" ", t)
    t = _PUNCT.sub(" ", t)
    return _SPACE.sub(" ", t).strip()


def pieces(text):
    """1장과 같은 두 글자 조각 집합(정규화한 뒤)."""
    out = set()
    for w in normalize(text).split():
        if len(w) == 1:
            out.add(w)
        for i in range(len(w) - 1):
            out.add(w[i:i + 2])
    return out


def jaccard(a, b):
    A, B = pieces(a), pieces(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def jaccard_detail(a, b):
    A, B = pieces(a), pieces(b)
    return {"a": sorted(A), "b": sorted(B), "shared": sorted(A & B), "union": len(A | B),
            "inter": len(A & B), "j": round(len(A & B) / len(A | B), 4)}


def pair_text(it):
    return it["q"] + " " + it["a"]


# ---------- 단계별 정제 ----------

def drop_empty(items):
    keep, drop = [], []
    for it in items:
        (drop if it["a"].strip() in PLACEHOLDERS else keep).append(it)
    return keep, drop


def is_volatile(answer):
    """정책 낱말과 숫자가 함께 있으면 자주 바뀌는 값으로 본다. '면 100%'처럼 숫자만 있는 답은 남는다."""
    return bool(VOLATILE_NUM.search(answer) and VOLATILE_KEY.search(answer))


def drop_volatile(items):
    keep, drop = [], []
    for it in items:
        (drop if is_volatile(it["a"]) else keep).append(it)
    return keep, drop


def drop_exact(items):
    """정규화한 (문의, 답)이 같으면 먼저 나온 것만 남긴다."""
    seen, keep, drop = {}, [], []
    for it in items:
        k = (normalize(it["q"]), normalize(it["a"]))
        if k in seen:
            drop.append({**it, "dup_of": seen[k]})
        else:
            seen[k] = it["id"]
            keep.append(it)
    return keep, drop


def near_pairs(items, threshold):
    out = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            s = jaccard(pair_text(items[i]), pair_text(items[j]))
            if s >= threshold:
                out.append((items[i]["id"], items[j]["id"], round(s, 4)))
    return out


def clusters(ids, pairs):
    """유니온 파인드. 묶음마다 id를 정렬해 돌려주고, 묶음은 첫 id 순으로 정렬한다."""
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in sorted(pairs):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups = {}
    for i in sorted(ids):
        groups.setdefault(find(i), []).append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def drop_near(items, threshold):
    """거의 중복 묶음마다 id가 가장 앞선 것 하나만 남긴다."""
    pairs = near_pairs(items, threshold)
    groups = clusters([it["id"] for it in items], pairs)
    keep_ids = {g[0] for g in groups}
    keep = [it for it in items if it["id"] in keep_ids]
    drop = [{**it, "near_of": next(g[0] for g in groups if it["id"] in g)} for it in items if it["id"] not in keep_ids]
    return keep, drop, pairs


def leak_scores(items, eval_items):
    """학습 후보마다 평가 문의 40건 가운데 가장 닮은 것과 그 점수."""
    out = []
    for it in items:
        best = sorted(((round(jaccard(it["q"], e["text"]), 4), e["id"]) for e in eval_items), key=lambda t: (-t[0], t[1]))[0]
        out.append({"id": it["id"], "q": it["q"], "eval_id": best[1], "score": best[0]})
    return out


def drop_leaks(items, eval_items, threshold):
    sc = {s["id"]: s for s in leak_scores(items, eval_items)}
    keep = [it for it in items if sc[it["id"]]["score"] < threshold]
    drop = [{**it, "eval_id": sc[it["id"]]["eval_id"], "score": sc[it["id"]]["score"]} for it in items if sc[it["id"]]["score"] >= threshold]
    return keep, drop


# ---------- 나누기 ----------

def split_seeded(items, val_n, seed):
    """random.Random(seed)로 섞은 뒤 앞의 val_n개를 검증으로. 같은 seed면 늘 같은 결과."""
    order = sorted(items, key=lambda it: it["id"])
    random.Random(seed).shuffle(order)
    val = sorted(order[:val_n], key=lambda it: it["id"])
    train = sorted(order[val_n:], key=lambda it: it["id"])
    return train, val


def split_hash(items, val_ratio, salt="ch03"):
    """id의 sha256으로 나누기. 데이터가 늘어도 이미 있던 항목은 같은 쪽에 남는다."""
    train, val = [], []
    for it in sorted(items, key=lambda it: it["id"]):
        h = int(hashlib.sha256((salt + it["id"]).encode()).hexdigest(), 16) % 10000
        (val if h < val_ratio * 10000 else train).append(it)
    return train, val


def cross_twins(train, val, threshold):
    """검증 항목마다 학습 세트에 닮은꼴(자카드 >= threshold)이 있는지."""
    out = []
    for v in val:
        s, tid = max(((jaccard(pair_text(v), pair_text(t)), t["id"]) for t in train), key=lambda x: (x[0], x[1]))
        out.append({"val": v["id"], "train": tid, "score": round(s, 4), "twin": s >= threshold})
    return out


# ---------- 가짜 모델: 외워서 답하기 ----------

def fake_memorizer(train, val, pass_at=0.8):
    """검증 문의와 가장 닮은 학습 문의를 찾아 그 답을 그대로 내는 가짜 모델.
    낸 답과 정답의 자카드가 pass_at 이상이면 '맞혔다'고 센다. 학습은 하지 않는다."""
    rows = []
    for v in val:
        s, t = max(((jaccard(v["q"], t["q"]), t["id"]) for t in train), key=lambda x: (x[0], x[1]))
        ans = next(x["a"] for x in train if x["id"] == t)
        j = jaccard(ans, v["a"])
        rows.append({"val": v["id"], "nearest": t, "q_sim": round(s, 4), "a_sim": round(j, 4), "pass": j >= pass_at})
    return {"n": len(rows), "passed": sum(r["pass"] for r in rows), "rows": rows}
