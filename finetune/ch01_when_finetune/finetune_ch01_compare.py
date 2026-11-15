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
# 문의의 실패 종류(data/inquiries.json의 note 기준): K 모르는 사실(우리 정책), F 정해진 형식(인사·서명·길이), T 우리 말투, I 지켜야 할 규칙 위반, R 여러 단계 계산, OK 문제없음
TYPES = ["K", "F", "T", "I", "R", "OK"]
# 방법 묶음: 빈 문자열은 아무것도 안 함, P 프롬프트, R 검색, F 파인튜닝와 그 조합
COMBOS = ["", "P", "R", "F", "PR", "PF", "RF", "PRF"]


def load(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def load_rules():
    return json.loads(RULE_FILE.read_text(encoding="utf-8"))


# ---------- 토큰 예산 ----------

def tokens_per_call(s, k, e, q):
    """지시문 s + 예시 k개 × e + 질문 q."""
    # 한 번의 호출 토큰 = 지시문 s + 예시 k개 x 예시당 e + 질문 q
    return s + k * e + q


def k_max(window, s, e, q, a):
    """문맥 창 window에 답 자리 a를 남기고 넣을 수 있는 예시 수의 최댓값."""
    # 창에서 지시문, 질문, 답변 예약 자리를 빼고 남은 칸
    room = window - s - q - a
    # 정수 나눗셈으로 내림해 예시 수를 구한다. 남는 칸이 음수면 0
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
        # 파인튜닝된 모델이 외운 정책(학습 때 문서)과 지금 정책을 따로 들고 있어, 정책이 바뀐 뒤의 낡은 답을 판정할 수 있다
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
        # 지난 답변에 그 사실이 min_fact_count번 이상 나왔을 때만 파인튜닝이 배운 것으로 본다(규칙 파일의 가정값)
        return self.past["fact_counts"][fact] >= self.rules["params"]["min_fact_count"]

    def ft_stale(self, item):
        fact = self.fact_of[item["gold"]]
        return self.trained_docs[fact] != self.now_docs[fact]

    def outcome(self, item, combo):
        """'ok' | 'fail' | 'stale'(옛 정책으로 자신 있게 틀림)."""
        t = item["failure"]
        # 이미 맞는 문의는 어떤 방법 묶음에서도 ok
        if t == "OK":
            return "ok"
        rules = self.rules["rules"]
        if t == "K":
            # 지식 문의(K): 검색이 정답 문서를 1등으로 가져오면 해결. 가져온 문서와 모델 기억이 충돌하면 문서를 따른다고 가정한 것(모형의 선택)
            if "R" in combo and self.retrieved(item) == item["gold"]:
                return "ok"  # 충돌 규칙: 붙인 문서를 따른다고 가정
            if "F" in combo and self.ft_knows(item):
                # 파인튜닝이 사실을 배웠어도, 그 사실의 정책 문구가 학습 이후 바뀌었으면 옛 값으로 자신 있게 틀린다(stale)
                return "stale" if self.ft_stale(item) else "ok"
            return "fail"
        for m in combo:
            # 규칙 표에서 방법 m이 실패 종류 t를 고치는지: True, False, 또는 계산으로 정하는 값("budget" 등)
            fx = rules[m][t]["fix"]
            if fx is True:
                return "ok"
            # budget이면 필요한 예시 수가 문맥 창에 들어가는지(k_max) 계산으로 판정
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
        # 남는 실패 = 틀린 것 + 옛 값으로 틀린 것
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
    # 전체에서 방법 묶음이 고치는 건수(종류별 실제 개수를 넘지 않게)를 뺀다
    return sum(c.values()) - sum(min(fixes.get(t, 0), c[t]) for t in c)
