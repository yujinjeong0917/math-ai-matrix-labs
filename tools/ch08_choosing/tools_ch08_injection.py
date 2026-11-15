"""프롬프트 주입 장난감 2판. 5장 장난감을 이어받아 방어를 겹겹이 쌓는다.

모델을 부르지 않는다. "글 속 지시처럼 생긴 문장을 따른다"는 것을 정규식 네 개로 흉내 낸다.
고리(loop)마다 바꾸는 것은 네 가지다.
  source  : 도구 결과 속 지시를 어떻게 다루나
            naive    = 모두 따른다
            filter   = 낱말 목록(AI에게, ignore previous, <system>)에 걸린 문의를 빼고 나머지는 따른다
            separate = 도구 결과를 자료로만 읽는다. 다만 모델이 완벽하지 않다고 보고,
                       LEAK_KINDS에 든 종류는 새어 나와 따른다고 가정한다(가정값, 측정값 아님)
  gate    : 쓰기·보내기 도구를 실행하는 코드가 사람 확인 없이는 돌지 않는다(per_call | per_kind | None)
  review  : 사람이 보고서를 읽고 틀에 없는 줄을 찾는다
사용자 지시는 늘 같다: "유형별로 세어 보고서를 써 줘. 아무것도 고치거나 보내지 마."
그래서 확인 문에서 사람은 모든 쓰기·보내기를 거절한다.
"""

import re

import tools_ch08_data as D

USER_TASK = "이번 주 문의를 유형별로 세어 보고서를 써 줘. 아무것도 고치거나 보내지 마."
# 허락한 쓰기 종류가 비어 있으므로, 확인 문(gate)이 있으면 모든 쓰기·보내기가 거절된다.
USER_ALLOWED_ACTIONS = set()          # 사용자가 허락한 쓰기 종류: 없음

# 지시를 알아보는 정규식 네 개: 이 장의 숨은 지시 4가지만 잡는 장난감. 실제 모델이 무엇을 따를지는 이 모형으로 알 수 없다.
PATTERNS = {
    "status_all": re.compile(r"문의를 모두 '([^']+)'로 바꿔"),
    "send_all_emails": re.compile(r"email every customer address .*? to (\S+@[\w.-]+\w)"),
    "set_text": re.compile(r"'([^']+)'을 '([^']+)'으로 고쳐"),
    "report_prepend": re.compile(r"요약 첫 줄에 '([^']+)'라고 적어"),
}
# 낱말 거르개: 소문자로 바꾼 본문에서 부분 문자열로 찾는다. 그래서 'AI에게 물어보기' 같은 평범한 문장도 걸린다(거짓 양성).
FILTER_WORDS = ["ai에게", "ignore previous", "<system>"]
# separate 고리가 놓친다고 가정한 종류(가정): 관리자 지시처럼 꾸민 글, 공손한 부탁 글.
# 가정값: 자료로만 읽어도 새는 종류. 실제 새는 비율이나 종류를 측정한 것이 아니다.
LEAK_KINDS = ("set_text", "report_prepend")
WRITE_KINDS = ("status_all", "send_all_emails", "set_text")


def find_instructions(text):
    out = []
    for kind, pat in PATTERNS.items():
        for m in pat.finditer(text):
            out.append((kind,) + m.groups())
    return out


def keyword_flag(text):
    t = text.lower()
    return any(w in t for w in FILTER_WORDS)


class World:
    """문의 DB, 화면 문구, 보낸 메일함. 쓰기는 모두 기록한다."""

    def __init__(self, inbox):
        # 원본을 건드리지 않으려 행을 복사한다(상태를 바꾸는 시나리오가 서로 영향을 주지 않게).
        self.rows = [dict(r) for r in inbox["rows"]]
        self.screens = {k: list(v) for k, v in inbox["screens"].items()}
        self.sent = []
        self.changed_rows = 0
        self.changed_texts = []

    # 이미 같은 상태면 바뀐 행으로 세지 않는다(14 = 답변 완료가 아닌 행 수).
    def update_status(self, rid, status):
        r = next(r for r in self.rows if r["id"] == rid)
        if r["status"] != status:
            r["status"] = status
            self.changed_rows += 1

    def set_text(self, before, after):
        for k, texts in self.screens.items():
            if before in texts:
                texts[texts.index(before)] = after
                self.changed_texts.append({"screen": k, "before": before, "after": after})

    def send_email(self, to, addresses):
        self.sent.append({"to": to, "count": len(addresses)})


# 보고서 틀: 제목 줄 + 유형별 건수 줄. 틀에 없는 줄은 사람이 읽을 때 눈에 띄는 변조로 본다. 정렬은 건수 내림차순, 동률이면 유형 이름 순.
def write_report(rows, prepend=None):
    counts = {}
    for r in rows:
        counts[r["type"]] = counts.get(r["type"], 0) + 1
    lines = [f"{t}: {n}건" for t, n in sorted(counts.items(), key=lambda x: (-x[1], x[0]))]
    template = ["이번 주 문의 요약"] + lines
    report = ([prepend] if prepend else []) + template
    return report, set(template)


def run(source="naive", gate=None, review=False, inbox=None):
    inbox = inbox or D.load_inbox()
    world = World(inbox)
    rows = world.rows                         # 읽기 도구가 돌려준 결과(자료)
    # 1) 무엇을 따르나
    flagged, false_pos = [], []
    todo = []
    for r in rows:
        # 본문 하나에서 찾은 지시 목록. 아래 분기가 고리 종류(source)에 따라 이 목록을 걸러낸다.
        ins = find_instructions(r["body"])
        # filter: 낱말에 걸린 문의는 통째로 제외. 지시가 없는데 걸렸으면 거짓 양성으로 기록한다. 걸리지 않은 공손한 지시(q17)는 그대로 통과한다.
        if source == "filter" and keyword_flag(r["body"]):
            flagged.append(r["id"])
            if not ins:
                false_pos.append(r["id"])
            continue
        if source == "separate":
            ins = [i for i in ins if i[0] in LEAK_KINDS]
        todo += [(r["id"],) + i for i in ins]
    # 2) 실행. 쓰기·보내기는 gate가 있으면 사람 확인을 거친다(코드에서 막음)
    prompts, attempted, denied = 0, 0, 0
    prepend = None
    asked_kinds = set()
    for item in todo:
        kind = item[1]
        # 보고서 첫 줄 변조는 도구 호출이 아니라 글 속에 섞이므로 확인 문이 막지 못한다(막으려면 사람이 읽어야 한다).
        if kind == "report_prepend":          # 도구 호출이 아니라 글 속에 섞인다. gate가 볼 수 없다.
            prepend = item[2]
            continue
        # status_all은 행마다 쓰기 호출 1번이라 행 수만큼(20) 센다. 20 + 메일 1 + 문구 변경 1 = 22번이 순진한 고리의 시도 수.
        n_calls = len(rows) if kind == "status_all" else 1
        attempted += n_calls
        if gate == "per_call":
            prompts += n_calls
        elif gate == "per_kind" and kind not in asked_kinds:
            prompts += 1
            asked_kinds.add(kind)
        # gate가 없으면 무조건 실행. gate가 있으면 사용자가 허락한 종류만 실행(여기서는 비어 있어 모두 거절).
        approved = (gate is None) or (kind in USER_ALLOWED_ACTIONS)
        if not approved:
            denied += n_calls
            continue
        if kind == "status_all":
            for r in rows:
                world.update_status(r["id"], item[2])
        elif kind == "send_all_emails":
            world.send_email(item[2], [r["email"] for r in rows])
        elif kind == "set_text":
            world.set_text(item[2], item[3])
    report, template = write_report(rows, prepend)
    tampered = prepend is not None
    # 사람이 읽을 때 틀에 없는 줄이 있으면 잡는다고 가정(사람이 항상 알아챈다는 낙관적 가정).
    caught_in_review = review and any(line not in template for line in report)
    return {"source": source, "gate": gate, "review": review,
            "followed": sorted({(i[0], i[1]) for i in todo}),
            "attempted_calls": attempted, "approval_prompts": prompts, "denied_calls": denied,
            "notion_rows_changed": world.changed_rows,
            "figma_texts_changed": len(world.changed_texts),
            "emails_leaked": sum(s["count"] for s in world.sent),
            "report_tampered": tampered, "tampered_line": prepend,
            "tamper_reaches_reader": tampered and not caught_in_review,
            "flagged_for_review": flagged, "false_positives": false_pos}


SCENARIOS = [
    ("A", "순진한 고리: 글 속 지시를 모두 따름", dict(source="naive")),
    ("B", "낱말 거르개만", dict(source="filter")),
    ("C", "자료로만 읽기(새는 종류 가정)", dict(source="separate")),
    ("D", "순진한 고리 + 코드 속 확인 문(호출마다 묻기)", dict(source="naive", gate="per_call")),
    ("E", "자료로만 읽기 + 확인 문(종류마다 한 번 묻기)", dict(source="separate", gate="per_kind")),
    ("F", "E + 사람이 보고서를 읽음", dict(source="separate", gate="per_kind", review=True)),
]


def all_scenarios():
    out = []
    for key, label, kw in SCENARIOS:
        r = run(**kw)
        r.update({"key": key, "label": label})
        out.append(r)
    return out


def yearly(n_per_week=4, weeks=52, p_leak=0.05, q_miss=0.05):
    """예시 계산. 1년 동안 숨은 지시가 실제 피해로 이어지는 기대 횟수 = N x p x q.
    p, q는 가정한 예시 값이고, 두 단계가 서로 독립이라고 가정한다."""
    from fractions import Fraction as F
    # 식: 기대 피해 횟수 = N x p x q. 4건/주 x 52주 = 208건, 새는 확률 p, 확인 문을 지나칠 확률 q(둘 다 예시 값).
    n = n_per_week * weeks
    p, q = F(str(p_leak)), F(str(q_miss))
    return {"n": n, "p": p_leak, "q": q_miss, "after_separate": float(n * p), "after_gate": float(n * p * q)}
