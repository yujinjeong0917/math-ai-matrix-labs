"""도구 선택 점검표. 팀의 조건으로 후보 구성마다 질문 10개를 판정한다.

판정은 통과 / 막힘 / 확인 필요 / 해당 없음 넷뿐이다. 점수를 매기거나 후보끼리 줄 세우지 않는다.
"""

import json

import tools_ch08_cost as C
import tools_ch08_data as D

CHECKLIST_FILE = D.DATA / "checklist.json"
PASS, BLOCK, CHECK, NA = "통과", "막힘", "확인 필요", "해당 없음"


def load_checklist():
    return json.loads(CHECKLIST_FILE.read_text(encoding="utf-8"))


def policy_row(policies, key):
    prov, prod = key.split("|")
    return next(r for r in policies["rows"] if r["provider"] == prov and r["product"] == prod)


def judge(rule, s, team, pol, up):
    if rule == "trains_default":
        if not team["personal_data"]:
            return NA, "개인정보가 없음"
        v = pol["trains_default"]
        if v == "no":
            return PASS, "학습에 쓰지 않는 것이 기본이라고 문서가 적음"
        if v == "yes":
            return BLOCK, "학습에 씀"
        return CHECK, "설정에 따라 다름. 설정 값을 직접 확인하고 기록해야 함"
    if rule == "retention":
        d = s["retention_days"]
        if d is None:
            return CHECK, "보관 기간을 이번에 확인하지 못함"
        if d <= team["max_retention_days"]:
            return PASS, f"기본 {d}일(정책 위반 표시 대화는 더 길 수 있음)"
        return BLOCK, f"{d}일"
    if rule == "region":
        if team["region_required"] is None:
            return PASS, "팀의 지역 요구가 없음(생기면 다시 판정)"
        return CHECK, "지역 지정 조건과 추가 요금 확인"
    if rule == "budget":
        key = s["cost_month"]
        if key is None:
            return CHECK, s.get("cost_note", "비용 계산 없음")
        if key == "ch06_no_usd":
            return CHECK, "6장은 달러 대신 끝까지 옳을 확률과 시도 수를 셈. 한 달 API 비용은 따로 계산해야 함"
        m = C.r4(C.month(up[key]))
        return (PASS if m <= team["budget_month_usd"] else BLOCK), f"한 달 {m}달러(앞 장 계산)"
    if rule == "least_tools":
        extra = [t for t in s["write_tools"] if t not in team["writes_needed"]]
        if extra:
            return BLOCK, "필요 없는 쓰기 도구: " + ", ".join(extra)
        missing = [t for t in team["writes_needed"] if t not in s["write_tools"]]
        if missing:
            return PASS, "넘치는 쓰기 도구는 없음. 대신 " + ", ".join(missing) + "는 사람이 직접 함"
        return PASS, "필요한 쓰기 도구만 있음"
    if rule == "separate":
        if not team["untrusted_input"]:
            return NA, "믿을 수 없는 글이 없음"
        return (PASS, "도구 결과로 따로 넘김") if s["untrusted_marked"] else (CHECK, "붙여 넣은 글이 지시와 한 덩어리로 섞임")
    if rule == "trifecta":
        three = [team["personal_data"], team["untrusted_input"], s["external_send"]]
        return (BLOCK, "셋이 모두 있음") if all(three) else (PASS, f"셋 가운데 {sum(three)}개")
    if rule == "gate":
        if not s["write_tools"]:
            return NA, "쓰기 도구가 없음"
        return (PASS, "실행 코드가 확인을 기다림") if s["gate"] else (BLOCK, "확인 없이 바로 실행")
    if rule == "review":
        if not team["human_reads_report"]:
            return NA, "팀이 요구하지 않음"
        return (PASS, "사람이 읽고 보냄") if s["human_reads"] else (BLOCK, "읽지 않고 바로 보냄")
    if rule == "dated":
        return (PASS, "가격표·약관 파일 이름에 날짜") if s["dated"] else (CHECK, "확인한 날짜 기록이 없음")
    raise ValueError(rule)


def evaluate():
    cl = load_checklist()
    pol = D.load_policies()
    up = D.load_upstream()
    out = []
    for s in cl["setups"]:
        p = policy_row(pol, s["provider_row"])
        answers = []
        for q in cl["questions"]:
            v, why = judge(q["rule"], s, cl["team"], p, up)
            answers.append({"id": q["id"], "verdict": v, "why": why})
        tally = {k: sum(1 for a in answers if a["verdict"] == k) for k in (PASS, BLOCK, CHECK, NA)}
        out.append({"id": s["id"], "label": s["label"], "answers": answers, "tally": tally,
                    "blocked_by": [a["id"] for a in answers if a["verdict"] == BLOCK]})
    return out
