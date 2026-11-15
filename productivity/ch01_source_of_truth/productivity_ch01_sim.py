"""생산성 도구 1장 모형: 복사 방식과 원본 등록부 방식에 같은 변경 카드를 넣고 어긋난 칸을 센다.

- 복사 방식: 값 하나가 세 자리에 따로 적혀 있다. 카드를 받은 사람은 자기 자리의 칸 하나만 고친다.
- 원본 등록부 방식: 값마다 원본 자리 하나. 고치는 건 원본에서만. 링크·임베드로 이은 자리는 저절로 따라오고,
  손으로 옮기는 자리는 담당이 원본을 보고 값 전체를 옮겨 적는다(확률 q로 성공, 기본 결정 실행에서는 하지 않음).

표준 라이브러리만 쓴다. NumPy 대응 구현은 productivity_ch01_np.py.

무엇을 재는가: 어긋난 칸 수(자리 x 값, 최대 9칸), 진짜 값이 어느 자리에도 안 남은 값의 수, 한 번도 진짜였던 적 없는 값이 든 칸 수,
변경 1건당 고칠 자리 수. 입력은 productivity_ch01_project의 카드·원본 등록부, 출력은 dict 또는 정수.
테스트가 확인하는 숫자(어긋난 칸 7/9, 등록부 방식 4칸, 편집 24회 대 19회, 0.81^8 = 0.1853)이다(장 본문 숫자와의 대응은 README로 확인 필요).
모형의 단순화: 사람은 카드에 적힌 칸 하나만 고치고(칸 단위), 손으로 옮기는 자리는 확률 q로 옮긴다. 현실의 부분 복사·문구 변형은 다루지 않는다.
"""

import copy
import random

import productivity_ch01_project as P


# ---------------------------------------------------------------- 결정적 실행(카드 8장)

def _truth_history(cards, initial=None):
    """카드를 순서대로 넣었을 때 값마다 '진짜 값'이 거쳐 간 모든 상태."""
    truth = copy.deepcopy(P.INITIAL if initial is None else initial)  # 깊은 복사: 원본 INITIAL을 바꾸지 않는다
    hist = {v: [dict(truth[v])] for v in truth}  # 값마다 [초기값, 변경 1 뒤, 변경 2 뒤, ...]
    for c in cards:
        truth[c["value"]][c["field"]] = c["new"]
        hist[c["value"]].append(dict(truth[c["value"]]))
    return truth, hist


def run_copy(cards, propagate=None, initial=None):
    """복사 방식. propagate(card, place) -> bool 이 주어지면 다른 자리에도 같은 칸을 옮겨 적는다(사람이 맞춰 줌).

    돌려주는 값: 최종 상태, 진짜 값, 카드마다 어긋난 칸 수, 자리별 마지막으로 고친 카드 번호.
    """
    init = P.INITIAL if initial is None else initial
    state = {p: copy.deepcopy(init) for p in P.PLACES}
    truth = copy.deepcopy(init)
    per_step, last_edit = [], {p: -1 for p in P.PLACES}
    for i, c in enumerate(cards):
        v, f, new = c["value"], c["field"], c["new"]
        truth[v][f] = new  # 진짜 값은 카드가 바꾸는 즉시 갱신된다
        for p in P.PLACES:
            if p == c["where"] or (propagate is not None and propagate(c, p)):
                if state[p][v][f] != new:  # 값이 실제로 바뀐 경우만 "마지막 수정"으로 친다(같은 값을 덮어쓴 것은 제외)
                    last_edit[p] = i
                state[p][v][f] = new
        per_step.append(mismatched(state, truth))
    return {"state": state, "truth": truth, "per_step": per_step, "last_edit": last_edit}


def run_register(cards, hand_ok=None, initial=None, source=None, links=None):
    """원본 등록부 방식. 카드가 어디서 왔든 원본 자리에서 고친다.

    hand_ok(card, place) -> bool: 손으로 옮기는 자리를 이번 변경 때 옮겼나. None이면 아무도 옮기지 않는다
    ("한 곳만 고친다" 규칙을 그대로 지킨 결과).
    """
    init = P.INITIAL if initial is None else initial
    source = P.SOURCE if source is None else source
    links = P.LINKS if links is None else links
    state = {p: copy.deepcopy(init) for p in P.PLACES}
    truth = copy.deepcopy(init)
    per_step = []
    for c in cards:
        v, f, new = c["value"], c["field"], c["new"]
        truth[v][f] = new
        src = source[v]
        state[src][v][f] = new  # 카드를 받은 자리가 어디든 원본 자리의 칸을 고친다
        for p, how in links[v].items():
            if how == "ref":
                state[p][v] = dict(state[src][v])            # 링크·임베드: 원본을 그대로 보여 준다
            elif hand_ok is not None and hand_ok(c, p):
                state[p][v] = dict(state[src][v])            # 손으로 옮길 때는 원본의 값 전체를 옮긴다
        per_step.append(mismatched(state, truth))
    return {"state": state, "truth": truth, "per_step": per_step}


def mismatched(state, truth):
    """진짜 값과 다른 칸 수(자리 x 값, 최대 9)."""
    return sum(state[p][v] != truth[v] for p in state for v in truth)


def never_true_cells(state, cards, initial=None):
    """한 번도 진짜였던 적이 없는 값이 들어 있는 칸 수. 칸 단위로 엇갈려 고치면 생긴다."""
    _, hist = _truth_history(cards, initial)
    return sum(state[p][v] not in hist[v] for p in state for v in hist)


def truth_anywhere(state, truth):
    """값마다 진짜 값이 세 자리 중 어디에라도 남아 있나."""
    return {v: any(state[p][v] == truth[v] for p in state) for v in truth}


def distinct_values(state):
    """값마다 세 자리에 서로 다른 값이 몇 종류 있는지(3이면 셋 다 다름). dict를 정렬된 튜플로 바꿔 집합에 넣는다."""
    return {v: len({tuple(sorted(state[p][v].items())) for p in state}) for v in P.VALUES}


# ---------------------------------------------------------------- 어느 쪽이 맞나: 나중에 고르는 규칙

def pick_majority(state, v):
    """두 자리 이상이 같은 값이면 그 값, 아니면 None(못 고름)."""
    vals = [state[p][v] for p in P.PLACES]  # 값은 dict라 list.count가 동등 비교로 센다
    for x in vals:
        if vals.count(x) >= 2:
            return x
    return None


def pick_latest_file(state, last_edit, v):
    """파일 수정 시각으로 고르기: 가장 최근에 수정된 파일의 값을 믿는다. 칸마다 시각이 없으니 파일 단위다."""
    # 동점이면 PLACES 순서(notion, figma, sheets)의 앞쪽이 뽑힌다. 아무것도 안 고친 자리는 -1이다
    p = max(P.PLACES, key=lambda x: last_edit[x])
    return state[p][v]


def pick_rule_place(state, v):
    """지금 정한 결정 규칙(숫자는 Sheets, 문장은 Notion)의 자리를 뒤늦게 믿기."""
    return state[P.SOURCE[v]][v]


def judge(run):
    """세 가지 사후 선택 규칙(다수결, 최근 수정 파일, 뒤늦게 정한 규칙 자리)이 값 3개 중 몇 개를 맞히는지 센다.
    right 맞힘, wrong 틀린 값을 골랐음, undecided 고르지 못함."""
    s, t = run["state"], run["truth"]
    out = {}
    for name, f in (("majority", lambda v: pick_majority(s, v)),
                    ("latest_file", lambda v: pick_latest_file(s, run["last_edit"], v)),
                    ("rule_place_after_the_fact", lambda v: pick_rule_place(s, v))):
        picks = {v: f(v) for v in P.VALUES}
        out[name] = {"right": sum(picks[v] == t[v] for v in P.VALUES),
                     "wrong": sum(picks[v] is not None and picks[v] != t[v] for v in P.VALUES),
                     "undecided": sum(picks[v] is None for v in P.VALUES)}
    return out


# ---------------------------------------------------------------- 편집 수

def edits_per_card(card, mode):
    """변경 1건을 끝까지 맞추려면 고쳐야 하는 자리 수."""
    if mode == "copy":
        return len(P.PLACES)                               # 세 자리 모두 손으로
    # ref(링크·임베드)는 편집이 필요 없고, hand만 사람이 옮겨 적어야 한다
    hand = sum(how == "hand" for how in P.LINKS[card["value"]].values())
    return 1 + hand                                        # 원본 1곳 + 손으로 옮기는 자리


# ---------------------------------------------------------------- 무작위 카드(몬테카를로)

FIELDS = {"price": ["won"], "hours": ["open", "close"], "cancel": ["free_before_h"]}


def random_cards(rng, m=8):
    """값·칸·새 값·고친 자리를 무작위로 고른 카드 m장. 새 값은 지금 진짜 값과 늘 다르다."""
    truth = copy.deepcopy(P.INITIAL)
    out = []
    for i in range(m):
        v = rng.choice(P.VALUES)
        f = rng.choice(FIELDS[v])
        new = truth[v][f] + rng.choice([-2, -1, 1, 2])  # 0을 뺀 증감이라 새 값은 항상 현재 값과 다르다
        truth[v][f] = new
        out.append({"id": f"R{i+1}", "value": v, "field": f, "new": new, "where": rng.choice(P.PLACES)})
    return out


def simulate(mode, q, rng, m=8, correlated=False):
    """한 번 돌린 결과. 복사 방식은 고친 사람이 다른 두 자리를 각각 확률 q로 맞춘다(correlated면 둘을 함께)."""
    cards = random_cards(rng, m)
    if mode == "copy":
        if correlated:
            # 카드 하나에 대해 판정을 한 번만 뽑아 두 자리에 같이 적용한다(함께 맞추거나 함께 놓친다)
            memo = {}

            def prop(c, p):
                if c["id"] not in memo:
                    memo[c["id"]] = rng.random() < q
                return memo[c["id"]]
        else:
            def prop(c, p):
                return rng.random() < q
        r = run_copy(cards, prop)
    elif mode == "register":
        r = run_register(cards, lambda c, p: rng.random() < q)
    elif mode == "register_ideal":
        links = {v: {p: "ref" for p in P.LINKS[v]} for v in P.LINKS}  # 이상적 경우: 모든 연결이 링크라 손 옮김이 필요 없다
        r = run_register(cards, None, links=links)
    else:
        raise ValueError(mode)
    anywhere = truth_anywhere(r["state"], r["truth"])
    return {
        "final_mismatch": r["per_step"][-1],
        "ever_clean": all(x == 0 for x in r["per_step"]),  # 한 번도 어긋난 적이 없나(매 단계 어긋난 칸 0)
        "final_clean": r["per_step"][-1] == 0,
        "exposure": sum(r["per_step"]),
        "never_true": never_true_cells(r["state"], cards),
        "values_lost": sum(not a for a in anywhere.values()),
        "majority_right": sum(pick_majority(r["state"], v) == r["truth"][v] for v in P.VALUES),
        "source_right": sum(r["state"][P.SOURCE[v]][v] == r["truth"][v] for v in P.VALUES),
    }


# ---------------------------------------------------------------- 일반 모형: 값 하나, 자리 k곳, 변경 m번

def generic_python(k, q, m, n, seed, correlated=False):
    """값 하나를 k곳에 둔다. 변경마다 고친 사람의 자리는 맞고, 나머지 k-1곳은 각각 확률 q로 맞춘다.
    새 값은 이전 값을 통째로 바꾼다. 돌려주는 값: (한 번도 안 어긋날 비율, 끝에 다 맞을 비율, 변경당 평균 어긋난 자리)."""
    rng = random.Random(seed)
    ever, final, wrong = 0, 0, 0  # 집계: 한 번도 안 어긋난 횟수, 마지막 변경에 다 맞은 횟수, 놓친 자리 총합
    for _ in range(n):
        ok_all = True
        for _ in range(m):
            if correlated:
                hit = rng.random() < q
                miss = 0 if hit else k - 1
            else:
                miss = sum(rng.random() >= q for _ in range(k - 1))
            wrong += miss
            ok_all = ok_all and miss == 0
        ever += ok_all
        # 주의: 여기서 miss는 안쪽 반복의 마지막 변경(m번째) 값이다(파이썬 변수가 반복문 밖에서도 남는 것을 이용)
        final += miss == 0
    return ever / n, final / n, wrong / (n * m)


def generic_exact(k, q, m, correlated=False):
    """같은 모형의 식. 독립이면 한 번도 안 어긋날 확률 (q^(k-1))^m, 마지막 변경만 보면 q^(k-1),
    변경당 어긋난 자리 기댓값 (k-1)(1-q). 함께 놓치는 경우(correlated)는 q^m, q, (k-1)(1-q)."""
    if k == 1:
        return 1.0, 1.0, 0.0
    if correlated:  # 변경마다 "전부 맞추기 / 전부 놓치기" 한 번의 동전이라 확률이 q ** (k - 1)이 아니라 q
        return q ** m, q, (k - 1) * (1 - q)
    return (q ** (k - 1)) ** m, q ** (k - 1), (k - 1) * (1 - q)


# ---------------------------------------------------------------- 등록부 점검

def lint_register(reg):
    """점검표. 문제 목록 [(항목 id, 문제)]을 돌려준다. 빈 목록이면 통과."""
    problems = []
    names = {}  # 항목 이름 -> 등록된 id 목록(같은 이름이 둘이면 원본이 둘이라는 뜻)
    for it in reg:
        names.setdefault(it["name"], []).append(it["id"])
        if it["tool"] != P.DECISION_RULE.get(it["kind"]):
            problems.append((it["id"], f"결정 규칙 위반: {it['kind']}인데 원본이 {it['tool']}"))
        if it["owner"] not in P.OWNERS:
            problems.append((it["id"], "담당 없음"))
        # 원본 수: 항목 자체의 도구 1곳 + 참조 목록에서 how가 "source"로 표시된 곳. 정확히 1이어야 한다
        srcs = [it["tool"]] + [t for t, _, how in it["refs"] if how == "source"]
        if len(srcs) != 1:
            problems.append((it["id"], f"원본이 {len(srcs)}곳"))
        for t, where, how in it["refs"]:
            if how == "hand" and not it.get("notify"):
                problems.append((it["id"], f"손으로 옮기는 {t}:{where}에 알림 방법 없음"))
            if how == "typed":
                problems.append((it["id"], f"{t}:{where}에 값을 다시 적음(등록부에 없는 사본)"))
            if t == it["tool"] and where == it["where"]:
                problems.append((it["id"], "원본 자리를 참조로도 적음"))
    for n, ids in names.items():
        if len(ids) > 1:
            problems.append((ids[1], f"'{n}' 항목이 {len(ids)}번 등록됨(원본 {len(ids)}개)"))
    return problems


def count_refs(reg):
    """등록부 전체에서 참조 방식별 개수(ref 링크·임베드 / hand 손 옮김)."""
    out = {"ref": 0, "hand": 0}
    for it in reg:
        for _, _, how in it["refs"]:
            if how in out:
                out[how] += 1
    return out
