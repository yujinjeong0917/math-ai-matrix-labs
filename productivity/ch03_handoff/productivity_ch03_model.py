"""생산성 도구 3장: 디자인을 개발자에게 넘길 때 무엇이 빠지나.

관통 프로젝트 "동네 스터디룸 예약"의 화면 4개를 "화면 x 상태 x 전환" 표로 만들고,
넘겨주는 방식(패키지)마다 개발자가 볼 수 있는 정보만 담아 질문 20개에 답하게 한다.
Figma를 띄우지 않는 모형이다. 표준 라이브러리만 쓴다.

채점 규칙(숫자를 보기 전에 정했다, README와 specs/question_rules.md에 같은 내용):
- 질문마다 필요한 정보 키(needs)가 있다. 패키지에 키가 모두 있으면 패키지의 값을 답으로 쓴다.
- 패키지에 키가 하나라도 없으면 "답할 수 없음"이다. 이때 개발자는 고정된 추측(guess)을 쓴다.
- 답을 진짜 의도(truth)와 비교해 네 갈래로 나눈다.
  correct(정보가 있고 맞음) / misread(정보는 있었는데 틀림) /
  guessed_right(정보가 없어 추측했는데 맞음) / guessed_wrong(정보가 없어 추측했는데 틀림)

입력: 질문 20개(QUESTIONS), 화면 x 상태 표(REQUIRED_STATES), 전환 목록, 넘겨주는 방식 다섯 가지의 패키지 dict.
출력: score가 돌려주는 네 갈래 개수와 질문별 분류, 완성도(그린 칸 / 필요한 칸), 질문 왕복 수. 단위는 개수.
모형의 단순화: 개발자의 추측은 질문마다 고정된 한 값이고, 답은 문자열·숫자의 정확 일치로만 채점한다(부분 점수 없음).
테스트의 기대값(상태 11칸, 이미지 방식 (3, 3, 2, 12), 체크리스트 20/20 등)이다(장 본문 숫자와의 대응은 README로 확인 필요).
"""

SCREENS = ["S01", "S02", "S03", "S04"]
SCREEN_NAME = {"S01": "목록", "S02": "상세", "S03": "예약", "S04": "완료"}
STATES = ["기본", "오류", "빈", "로딩"]

# 화면마다 필요한 상태. 나머지 칸은 "이 화면엔 없음"과 이유를 적어야 한다.
REQUIRED_STATES = {
    "S01": ["기본", "오류", "빈", "로딩"],
    "S02": ["기본", "오류", "로딩"],
    "S03": ["기본", "오류", "로딩"],
    "S04": ["기본"],
}
NOT_HERE_REASON = {
    ("S02", "빈"): "방을 눌러야 열리는 화면이라 방이 없을 수 없음",
    ("S03", "빈"): "입력 화면이라 비어 있는 상태가 곧 기본",
    ("S04", "오류"): "예약 확정 응답을 받은 뒤에만 열림(오류는 S03에서)",
    ("S04", "빈"): "보여 줄 예약이 항상 하나 있음",
    ("S04", "로딩"): "확정 응답과 함께 열려 따로 기다리지 않음",
}

# 전환(프로토타입 연결). (출발, 도착, 무엇을 할 때)
FULL_TRANSITIONS = [
    (("S01", "로딩"), ("S01", "기본"), "불러오기 성공"),
    (("S01", "로딩"), ("S01", "빈"), "결과 0개"),
    (("S01", "로딩"), ("S01", "오류"), "불러오기 실패"),
    (("S01", "오류"), ("S01", "로딩"), "다시 시도"),
    (("S01", "빈"), ("S01", "로딩"), "날짜 바꾸기"),
    (("S01", "기본"), ("S02", "로딩"), "방 누르기"),
    (("S02", "로딩"), ("S02", "기본"), "불러오기 성공"),
    (("S02", "로딩"), ("S02", "오류"), "불러오기 실패"),
    (("S02", "오류"), ("S02", "로딩"), "다시 시도"),
    (("S02", "오류"), ("S01", "기본"), "뒤로"),
    (("S02", "기본"), ("S01", "기본"), "뒤로(스크롤 위치 유지)"),
    (("S02", "기본"), ("S03", "기본"), "예약하기"),
    (("S03", "기본"), ("S02", "기본"), "뒤로"),
    (("S03", "기본"), ("S03", "로딩"), "예약 확정"),
    (("S03", "로딩"), ("S04", "기본"), "확정 성공"),
    (("S03", "로딩"), ("S03", "오류"), "시간 겹침"),
    (("S03", "오류"), ("S03", "기본"), "시간 다시 고르기(입력값 유지)"),
    (("S04", "기본"), ("S01", "로딩"), "목록으로(새로 불러오기)"),
]
START = ("S01", "로딩")  # 앱을 열면 목록을 불러오는 중부터 시작한다(checks.py의 도달 검사가 이 노드에서 출발한다)

# 메신저로 보낸 설명에 들어 있던 전환. 이미지에는 기본 상태만 있으니 기본끼리만 잇는다.
MESSENGER_TRANSITIONS = [
    (("S01", "기본"), ("S02", "기본"), "방 누르면 상세"),
    (("S02", "기본"), ("S03", "기본"), "예약하기 누르면 예약 화면"),
    (("S03", "기본"), ("S04", "기본"), "확정하면 완료"),
]

# 디자인 토큰: 값에 이름을 붙인 것(이름이 있어야 값이 바뀔 때 구현 쪽이 따라온다). 단위 px, 색은 16진수
TOKENS = {
    "space.4": 16,
    "space.2": 8,
    "size.button": 48,
    "font.title": 20,
    "font.body": 15,
    "color.primary": "#0B7A55",
    "color.text.subtle": "#6B7280",
}

# needs 키 형식: measure:치수 / font:글자 / color:색 / copy:문구 / state:화면:상태 / variant:변형 / edge:출발>도착 / hit:터치영역
# 질문 20개. needs: 답에 필요한 정보 키. truth: 디자이너의 의도. guess: 정보가 없을 때 개발자가 쓰는 고정 추측.
QUESTIONS = [
    {"id": "Q01", "cat": "치수", "screen": "S01", "q": "목록 셀 좌우 여백", "needs": ["measure:cell.pad"], "truth": 16, "guess": 16},
    {"id": "Q02", "cat": "치수", "screen": "S01", "q": "목록 셀 사이 간격", "needs": ["measure:cell.gap"], "truth": 8, "guess": 8},
    {"id": "Q03", "cat": "치수", "screen": "S02", "q": "큰 버튼 높이", "needs": ["measure:button.h"], "truth": 48, "guess": 48},
    {"id": "Q04", "cat": "글자", "screen": "S02", "q": "제목 글자 크기", "needs": ["font:title"], "truth": 20, "guess": 16},
    {"id": "Q05", "cat": "글자", "screen": "S02", "q": "본문 글자 크기", "needs": ["font:body"], "truth": 15, "guess": 16},
    {"id": "Q06", "cat": "색", "screen": "S01", "q": "버튼 색", "needs": ["color:primary"], "truth": "#0B7A55", "guess": "#00A86B"},
    {"id": "Q07", "cat": "색", "screen": "S03", "q": "안내 문구 색", "needs": ["color:subtle"], "truth": "#6B7280", "guess": "#999999"},
    {"id": "Q08", "cat": "문구", "screen": "S01", "q": "목록 버튼 문구", "needs": ["copy:list.button"], "truth": "예약", "guess": "예약"},
    {"id": "Q09", "cat": "문구", "screen": "S01", "q": "빈 목록 문구", "needs": ["state:S01:빈"], "truth": "조건에 맞는 방이 없어요. 날짜를 바꿔 보세요.", "guess": "데이터가 없습니다."},
    {"id": "Q10", "cat": "문구", "screen": "S01", "q": "목록 불러오기 실패 문구", "needs": ["state:S01:오류"], "truth": "목록을 불러오지 못했어요.", "guess": "오류가 발생했습니다."},
    {"id": "Q11", "cat": "문구", "screen": "S03", "q": "예약 시간 겹침 문구", "needs": ["state:S03:오류"], "truth": "이미 예약된 시간이에요. 다른 시간을 골라 주세요.", "guess": "오류가 발생했습니다."},
    {"id": "Q12", "cat": "상태", "screen": "S01", "q": "목록 불러오는 중 모습", "needs": ["state:S01:로딩"], "truth": "회색 셀 3개", "guess": "가운데 스피너"},
    {"id": "Q13", "cat": "상태", "screen": "S02", "q": "상세 불러오는 중 모습", "needs": ["state:S02:로딩"], "truth": "사진 자리 회색 상자", "guess": "가운데 스피너"},
    {"id": "Q14", "cat": "상태", "screen": "S03", "q": "예약 확정 누른 뒤 버튼", "needs": ["state:S03:로딩"], "truth": "비활성 + '예약 중…'", "guess": "가운데 스피너"},
    {"id": "Q15", "cat": "상태", "screen": "S03", "q": "필수 입력 전 예약 버튼", "needs": ["variant:S03.button.disabled"], "truth": "비활성(회색)", "guess": "활성, 누르면 오류"},
    {"id": "Q16", "cat": "동작", "screen": "S02", "q": "상세에서 뒤로 가면", "needs": ["edge:S02:기본>S01:기본"], "truth": "목록, 스크롤 위치 유지", "guess": "목록 맨 위"},
    {"id": "Q17", "cat": "동작", "screen": "S01", "q": "목록 불러오기 실패 뒤 할 수 있는 일", "needs": ["edge:S01:오류>S01:로딩"], "truth": "다시 시도 버튼", "guess": "없음(새로고침)"},
    {"id": "Q18", "cat": "동작", "screen": "S03", "q": "시간 겹침 오류 뒤 입력값", "needs": ["edge:S03:오류>S03:기본"], "truth": "유지", "guess": "유지"},
    {"id": "Q19", "cat": "동작", "screen": "S04", "q": "완료에서 목록으로 가면", "needs": ["edge:S04:기본>S01:로딩"], "truth": "목록 새로 불러오기", "guess": "목록 새로 불러오기"},
    {"id": "Q20", "cat": "접근성", "screen": "S02", "q": "뒤로 가기 버튼 터치 영역(px)", "needs": ["hit:back"], "truth": 44, "guess": 24},
]


def node_key(n):
    """(화면, 상태) 노드를 "S01:로딩" 같은 문자열로."""
    return f"{n[0]}:{n[1]}"


def edge_key(a, b):
    """간선 키 "edge:출발>도착". 질문의 needs와 패키지 info가 이 키로 만난다."""
    return f"edge:{node_key(a)}>{node_key(b)}"


def required_cells():
    """화면마다 꼭 필요한 (화면, 상태) 칸 11개."""
    return [(s, st) for s in SCREENS for st in REQUIRED_STATES[s]]


def completeness(drawn):
    """정의된 칸 수 / 필요한 칸 수. drawn은 (화면, 상태) 모음."""
    req = required_cells()
    have = sum(1 for c in req if c in set(drawn))  # 필요한 칸이 아닌 것을 그려도 분자에 안 센다
    return have, len(req)


def completeness_by_formula(drawn):
    """같은 값을 식으로: 분모 = sum_s |Sigma_s|, 분자 = sum_s |drawn_s ∩ Sigma_s|."""
    num = sum(len(set(st for (s2, st) in drawn if s2 == s) & set(REQUIRED_STATES[s])) for s in SCREENS)
    den = sum(len(REQUIRED_STATES[s]) for s in SCREENS)
    return num, den


def decided_cells(drawn, not_here):
    """4x4 칸 가운데 그렸거나 '이 화면엔 없음(이유)'을 적은 칸."""
    return sum(1 for s in SCREENS for st in STATES if (s, st) in set(drawn) or (s, st) in not_here)


# ---------------------------------------------------------------- 넘겨주는 방식(패키지)

def _measures(scale):
    """이미지에서 잰 값. scale배로 내보낸 이미지에서 픽셀을 세면 값이 scale배로 나온다."""
    return {
        "measure:cell.pad": TOKENS["space.4"] * scale,
        "measure:cell.gap": TOKENS["space.2"] * scale,
        "measure:button.h": TOKENS["size.button"] * scale,
    }


def _colors():
    """색 값을 있는 그대로 읽을 수 있는 정보(이미지에서도 색은 찍어 볼 수 있다는 가정)."""
    return {"color:primary": TOKENS["color.primary"], "color:subtle": TOKENS["color.text.subtle"]}


def _full_state_info():
    """상태·변형 질문의 진짜 답. 모든 상태를 그린 파일에서는 이 정보가 다 보인다고 가정한다."""
    info = {}
    for q in QUESTIONS:
        for k in q["needs"]:
            if k.startswith("state:") or k.startswith("variant:"):
                info[k] = q["truth"]
    return info


def _edges(transitions, behavior_truth):
    """전환 목록을 "간선 키 -> 동작 설명" 정보로 바꾼다. 동작 질문의 truth가 있으면 그 설명을, 없으면 "연결됨"만 안다."""
    info = {}
    for a, b, _ in transitions:
        k = edge_key(a, b)
        info[k] = behavior_truth.get(k, "연결됨")
    return info


def _behavior_truth():
    """동작 질문(cat == "동작")의 needs 키 -> truth."""
    return {q["needs"][0]: q["truth"] for q in QUESTIONS if q["cat"] == "동작"}


def package_image(scale_note):
    """이미지(2배로 내보낸 PNG) + 메신저 설명. scale_note가 False면 '2배'라는 말이 없다."""
    # 2배로 내보낸 PNG인데 "2배"라는 설명이 없으면 개발자가 32픽셀을 32px로 읽는다(실제 16px) -> 치수 3개가 misread
    read_scale = 1 if scale_note else 2  # 개발자가 픽셀을 CSS px로 읽는 배율 오차
    info = {}
    info.update(_measures(read_scale))
    info.update(_colors())  # PNG라 색 값은 그대로 찍어 볼 수 있다(토큰 이름은 없다)
    info["copy:list.button"] = "예약"
    if scale_note:
        info.update(_edges(MESSENGER_TRANSITIONS, {}))
    drawn = [(s, "기본") for s in SCREENS]
    return {"name": "이미지 + 메신저" + (" (배율 설명 있음)" if scale_note else " (배율 설명 없음)"),
            "info": info, "drawn": drawn, "not_here": {}, "transitions": MESSENGER_TRANSITIONS if scale_note else [],
            "token_names": False, "live": False}


def package_checklist():
    """체크리스트 + 프로토타입(보기 권한). 상태 11칸, 없음 5칸, 전환 18개, 토큰 이름, 터치 영역 행."""
    info = {}
    info.update(_measures(1))
    info["font:title"] = TOKENS["font.title"]
    info["font:body"] = TOKENS["font.body"]
    info.update(_colors())
    info["copy:list.button"] = "예약"
    info.update(_full_state_info())
    info.update(_edges(FULL_TRANSITIONS, _behavior_truth()))
    info["hit:back"] = 44
    return {"name": "체크리스트 + 프로토타입", "info": info, "drawn": required_cells(),
            "not_here": dict(NOT_HERE_REASON), "transitions": FULL_TRANSITIONS, "token_names": True, "live": False}


def package_devmode(file_complete):
    """Dev Mode로 파일을 직접 본다. 값은 정확하지만, 그려진 것만 보인다.

    file_complete=False: 기본 상태 4장만 그린 파일(실패 예제와 같은 파일) + 앞으로 가는 연결 3개.
    뒤로 가기 버튼은 아이콘 24px 프레임만 그려져 있어 inspect에도 24가 나온다.
    """
    info = {}
    info.update(_measures(1))
    info["font:title"] = TOKENS["font.title"]
    info["font:body"] = TOKENS["font.body"]
    info.update(_colors())
    info["copy:list.button"] = "예약"
    if file_complete:
        info.update(_full_state_info())
        info.update(_edges(FULL_TRANSITIONS, _behavior_truth()))
        info["hit:back"] = 44
        drawn, not_here, tr = required_cells(), dict(NOT_HERE_REASON), FULL_TRANSITIONS
    else:
        info.update(_edges(MESSENGER_TRANSITIONS, {}))
        info["hit:back"] = 24  # 그린 그대로: 아이콘 프레임 크기
        drawn, not_here, tr = [(s, "기본") for s in SCREENS], {}, MESSENGER_TRANSITIONS
    return {"name": "Dev Mode" + (" + 체크리스트를 채운 파일" if file_complete else " (기본 상태만 그린 파일)"),
            "info": info, "drawn": drawn, "not_here": not_here, "transitions": tr, "token_names": True, "live": True}


def all_packages():
    """비교 대상 다섯 패키지(이미지 2종, 체크리스트, Dev Mode 2종)."""
    return [package_image(False), package_image(True), package_checklist(),
            package_devmode(False), package_devmode(True)]


# ---------------------------------------------------------------- 채점

def answer(q, pkg):
    """(답, 정보가 있었나). needs가 여러 키면 모두 있어야 답할 수 있고, 답은 첫 키의 값이다."""
    info = pkg["info"]
    if all(k in info for k in q["needs"]):
        return info[q["needs"][0]], True
    return q["guess"], False


def classify(q, pkg):
    """네 갈래 분류. 정보가 있을 때만 correct/misread, 없을 때는 고정 추측의 맞고 틀림으로 나눈다."""
    val, had = answer(q, pkg)
    ok = val == q["truth"]
    if had:
        return "correct" if ok else "misread"
    return "guessed_right" if ok else "guessed_wrong"


def score(pkg):
    """패키지 하나의 네 갈래 개수와 질문별 분류. unanswerable = 추측으로 답한 수, wrong_total = 틀린 총합."""
    out = {"correct": 0, "misread": 0, "guessed_right": 0, "guessed_wrong": 0}
    per_q = {}
    for q in QUESTIONS:
        c = classify(q, pkg)
        out[c] += 1
        per_q[q["id"]] = c
    out["unanswerable"] = out["guessed_right"] + out["guessed_wrong"]
    out["wrong_total"] = out["misread"] + out["guessed_wrong"]
    out["per_question"] = per_q
    return out


def by_category(pkg):
    """질문 분류(치수·글자·색·문구·상태·동작·접근성)별 문항 수, 답할 수 없는 수, 틀린 수."""
    cats = {}
    for q in QUESTIONS:
        c = classify(q, pkg)
        d = cats.setdefault(q["cat"], {"n": 0, "unanswerable": 0, "wrong": 0})
        d["n"] += 1
        if c.startswith("guessed"):
            d["unanswerable"] += 1
        if c in ("misread", "guessed_wrong"):
            d["wrong"] += 1
    return cats


def question_rounds(pkg):
    """규칙: 개발자는 화면을 S01→S04 순서로 만들고, 답할 수 없는 질문이 하나라도 있는 화면마다
    질문을 한 번에 묶어 보낸다(왕복 1회). 질문 수 = 답할 수 없는 항목 수."""
    blocked = {}
    for q in QUESTIONS:
        if not answer(q, pkg)[1]:
            blocked.setdefault(q["screen"], []).append(q["id"])
    return {"questions": sum(len(v) for v in blocked.values()), "rounds": len(blocked),
            "by_screen": {s: blocked.get(s, []) for s in SCREENS}}


def designer_cost(pkg):
    """디자이너가 넘기기 전에 만들어야 하는 것의 개수(시간 대신 센다)."""
    return {"frames_drawn": len(pkg["drawn"]),
            "cells_decided": decided_cells(pkg["drawn"], pkg["not_here"]),
            "transitions": len(pkg["transitions"])}


# ---------------------------------------------------------------- 토큰 이름 vs 값 복사

BUTTON_SITES = 7  # 2장과 같은 버튼 7개(목록 4, 상세·예약·완료 각 1). 구현 코드에서 버튼 색을 쓰는 곳의 수로 가정


def token_change_follow(token_names, new_value="#0A6B4B"):
    """구현 코드가 버튼 색을 7곳에 쓴다. 토큰 이름으로 썼으면 토큰 값 한 곳을 바꾸면 따라오고,
    색 값을 복사해 썼으면 7곳이 옛 값으로 남는다."""
    tokens = {"color.primary": TOKENS["color.primary"]}
    sites = ["var:color.primary" if token_names else TOKENS["color.primary"] for _ in range(BUTTON_SITES)]
    tokens["color.primary"] = new_value  # 디자인 쪽에서 토큰 값 변경
    # "var:" 접두사(4글자)를 떼고 토큰 값을 찾는다. 복사한 색 값(접두사 없음)은 새 값이 반영되지 않는다
    rendered = [tokens[s[4:]] if s.startswith("var:") else s for s in sites]
    return {"follow": sum(v == new_value for v in rendered), "stale": sum(v != new_value for v in rendered),
            "edit_locations": 1 if token_names else BUTTON_SITES}
