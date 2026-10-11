"""같은 문의 세트에 프롬프트(P)·검색(R)·파인튜닝(F)을 대 보고, 남는 실패를 센다.

규칙은 data/fix_rules_2026-10-11.json에서 읽는다(입력). 계산으로 정해지는 칸은 셋이다.
- P × T: 필요한 예시 수가 문맥 창에 들어가는가(토큰 예산)
- R × K: 검색기가 정답 문서를 1등으로 가져오는가
- F × K: 지난 답변에 그 사실이 충분히 나왔는가, 정책이 바뀐 뒤에도 맞는가
"""

import json
from itertools import combinations
from pathlib import Path

import finetune_ch01_retrieve as RT

HERE = Path(__file__).parent
DATA = HERE / "data"
RULE_FILE = DATA / "fix_rules_2026-10-11.json"
TYPES = ["K", "F", "T", "I", "R", "OK"]
COMBOS = ["", "P", "R", "F", "PR", "PF", "RF", "PRF"]


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def load_rules():
    return json.loads(RULE_FILE.read_text(encoding="utf-8"))


# ---------- 토큰 예산 ----------

def tokens_per_call(s, k, e, q):
    """지시문 s + 예시 k개 × e + 질문 q."""
    return s + k * e + q


def k_max(window, s, e, q, a):
    """문맥 창 window에 답 자리 a를 남기고 넣을 수 있는 예시 수의 최댓값."""
    room = window - s - q - a
    return max(room // e, 0)


# ---------- 한 문의를 한 방법이 고치나 ----------

class Model:
    def __init__(self, policies, rules=None, past=None, budget=None, window=8192, trained_on=None):
        self.docs = policies["docs"]
        self.rules = rules or load_rules()
        self.past = past or load("past_answers.json")
        self.budget = budget or load("budget.json")
        self.window = window
        # 파인튜닝은 학습할 때의 정책(trained_on)을 외운다. 기본은 지금 정책과 같다.
        self.trained_docs = {d["fact"]: d["text"] for d in (trained_on or policies)["docs"]}
        self.now_docs = {d["fact"]: d["text"] for d in self.docs}
        self.fact_of = {d["id"]: d["fact"] for d in self.docs}

    def kmax(self):
        b = self.budget
        return k_max(self.window, b["system_tokens"], b["example_tokens"], b["question_tokens"], b["answer_reserve_tokens"])

    def retrieved(self, item):
        return RT.top1(item["text"], self.docs)

    def ft_knows(self, item):
        fact = self.fact_of[item["gold"]]
        return self.past["fact_counts"][fact] >= self.rules["params"]["min_fact_count"]

    def ft_stale(self, item):
        fact = self.fact_of[item["gold"]]
        return self.trained_docs[fact] != self.now_docs[fact]

    def outcome(self, item, combo):
        """'ok' | 'fail' | 'stale'(옛 정책으로 자신 있게 틀림)."""
        t = item["failure"]
        if t == "OK":
            return "ok"
        rules = self.rules["rules"]
        if t == "K":
            if "R" in combo and self.retrieved(item) == item["gold"]:
                return "ok"  # 충돌 규칙: 붙인 문서를 따른다고 가정
            if "F" in combo and self.ft_knows(item):
                return "stale" if self.ft_stale(item) else "ok"
            return "fail"
        for m in combo:
            fx = rules[m][t]["fix"]
            if fx is True:
                return "ok"
            if fx == "budget" and self.rules["params"]["needed_examples_T"] <= self.kmax():
                return "ok"
        return "fail"

    def tally(self, items, combo):
        out = {"ok": 0, "fail": 0, "stale": 0}
        by_type = {t: {"ok": 0, "fail": 0, "stale": 0} for t in TYPES}
        for it in items:
            o = self.outcome(it, combo)
            out[o] += 1
            by_type[it["failure"]][o] += 1
        out["remaining"] = out["fail"] + out["stale"]
        out["by_type"] = by_type
        return out


def counts(items):
    c = {t: 0 for t in TYPES}
    for it in items:
        c[it["failure"]] += 1
    return c


def hand_remaining(c, fixes):
    """손계산용: 실패 종류별 개수 c와, 방법 묶음이 고치는 개수 fixes(종류별)로 남는 수."""
    return sum(c.values()) - sum(min(fixes.get(t, 0), c[t]) for t in c)
