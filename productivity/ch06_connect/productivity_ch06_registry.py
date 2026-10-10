"""생산성 도구 6장 데이터: 관통 프로젝트 "동네 스터디룸 예약"의 기준 원본 등록부, 1주 변경 이벤트 12개, 권한 표, 자동화 후보.

값은 손으로 적었다. 실존 회사·인물 없음. 1장 등록부(정보 항목 / 원본 위치 / 참조처)를 6장 규칙 표로 넓힌 모양이다.
1장 실습 폴더는 이 장과 동시에 쓰이고 있어 import하지 않고, 필요한 항목을 여기 새로 적었다.

간선(원본 -> 참조처)의 갱신 방식은 세 가지다.
- "ref"   자동 참조: 링크, 임베드, 수식 참조, relation, 컴포넌트 인스턴스. 원본이 바뀌면 따라 바뀐다.
- "human" 사람 확인: 사람이 보고 손으로 옮겨 적는다. 놓칠 수 있다.
- "auto"  자동화: 사람 확인 간선 가운데 자동화 후보로 고른 것. 자동화가 살아 있으면 따라 바뀐다.
"""

# 위치 이름은 "도구:자리" 꼴. 도구는 figma / notion / sheets 셋.
# 항목마다 원본(src) 하나, 간선(edges) 여럿. 간선은 (보내는 곳, 받는 곳, 방식).
# 받는 곳이 다시 다른 곳에 값을 보낼 수 있다(예: Figma 컴포넌트 -> 인스턴스).
ITEMS = {
    "I01": {"name": "취소 규정 문구", "src": "notion:Specs/S02.취소규정",
            "edges": [("notion:Specs/S02.취소규정", "figma:Components/취소안내", "human"),
                      ("figma:Components/취소안내", "figma:S02_상세.취소안내", "ref"),   # 컴포넌트 인스턴스
                      ("figma:Components/취소안내", "figma:S04_완료.취소안내", "ref"),
                      ("notion:Specs/S02.취소규정", "sheets:metrics!안내", "ref")]},    # 문구 대신 Notion 링크
    "I02": {"name": "취소율 지표 정의", "src": "notion:Specs/지표정의.취소율",
            "edges": [("notion:Specs/지표정의.취소율", "sheets:metrics!B2수식", "human")]},  # 수식은 사람이 고친다(5장)
    "I03": {"name": "규정 변경 기준값", "src": "notion:Decisions/D04.기준",
            "edges": [("notion:Decisions/D04.기준", "sheets:metrics!threshold", "auto")]},  # 자동화 후보 A3
    "I04": {"name": "지점 목록", "src": "sheets:branches",
            "edges": [("sheets:branches", "sheets:clean!lookup", "ref"),
                      ("sheets:branches", "notion:Projects.지점수", "human")]},
    "I05": {"name": "운영 시간", "src": "notion:Specs/S02.운영시간",
            "edges": [("notion:Specs/S02.운영시간", "figma:S02_상세.운영시간", "human"),
                      ("notion:Specs/S02.운영시간", "sheets:capacity!hours", "human")]},
    "I06": {"name": "시간당 요금", "src": "sheets:lookup!price",
            "edges": [("sheets:lookup!price", "figma:S03_예약.요금", "human"),
                      ("sheets:lookup!price", "notion:Specs/S03.요금", "ref")]},  # 시트 임베드
    "I07": {"name": "예약 화면 프레임", "src": "figma:S03_예약",
            "edges": [("figma:S03_예약", "notion:Tasks.Figma링크", "ref")]},  # 링크는 그대로, 내용은 따라온다
    "I08": {"name": "D10 확정(전화 인증 철회)", "src": "notion:Decisions/D10.상태",
            "edges": [("notion:Decisions/D10.상태", "notion:Tasks/T17-T18", "auto"),   # 자동화 후보 A2
                      ("notion:Decisions/D10.상태", "figma:S03_예약.전화칸", "human")]},
    "I09": {"name": "예약 상태 표준값", "src": "sheets:lookup!status",
            "edges": [("sheets:lookup!status", "sheets:clean!status", "ref"),
                      ("sheets:lookup!status", "notion:Specs/지표정의.상태목록", "human")]},
    "I10": {"name": "예약 버튼 문구", "src": "notion:Specs/S03.버튼문구",
            "edges": [("notion:Specs/S03.버튼문구", "figma:Components/버튼", "human"),
                      ("figma:Components/버튼", "figma:S03_예약.버튼", "ref")]},
    "I11": {"name": "주간 취소율 값", "src": "sheets:metrics!B2값",
            "edges": [("sheets:metrics!B2값", "notion:Projects.주간리포트", "human")]},
    "I12": {"name": "런칭일", "src": "notion:Decisions/D09.날짜",
            "edges": [("notion:Decisions/D09.날짜", "notion:Tasks/T19-T20", "ref"),
                      ("notion:Decisions/D09.날짜", "sheets:metrics!기간", "human")]},
}

# 자동화 후보 3개(설계도 6장): 모두 사람 확인 간선이나 점검 결과를 보는 사람 자리에서 골랐다.
AUTOMATIONS = {
    "A1": {"what": "Sheets checks가 FALSE가 되면 알림", "tool": "Apps Script 설치형 트리거",
           "covers": "checks"},
    "A2": {"what": "Notion 결정 상태가 확정이 되면 관련 할 일 만들기 + 담당자 알림", "tool": "Notion 데이터베이스 자동화",
           "covers": ("I08", "notion:Tasks/T17-T18")},
    "A3": {"what": "Notion 결정의 기준값을 Sheets 기준 칸으로 옮기기", "tool": "API 스크립트(시간 트리거)",
           "covers": ("I03", "sheets:metrics!threshold")},
}

# 1주 변경 이벤트 12개. day = 1~7, where = 실제로 고친 곳(규칙이 없으면 손에 익은 곳에서 고친다),
# by = 고친 사람. stray = 원본이 아닌 곳(참조처)에서 고친 경우.
EVENTS = [
    {"id": "E01", "day": 1, "item": "I10", "what": "예약 버튼 문구 수정", "by": "민지", "where": "figma:Components/버튼"},
    {"id": "E02", "day": 1, "item": "I01", "what": "취소 규정 24시간 -> 12시간", "by": "서연", "where": "notion:Specs/S02.취소규정"},
    {"id": "E03", "day": 2, "item": "I02", "what": "취소율 정의를 지점 평균에서 전체로", "by": "서연", "where": "notion:Specs/지표정의.취소율"},
    {"id": "E04", "day": 2, "item": "I04", "what": "지점 B21 추가(가져오기 스크립트로 붙임)", "by": "가져오기 스크립트", "where": "sheets:branches",
     "via_script": True},
    {"id": "E05", "day": 3, "item": "I08", "what": "D10 확정", "by": "준호", "where": "notion:Decisions/D10.상태"},
    {"id": "E06", "day": 3, "item": "I05", "what": "운영 시간 22시 -> 23시(운영 담당이 화면 문구에서 바로 고침)", "by": "하린",
     "where": "figma:S02_상세.운영시간"},
    {"id": "E07", "day": 4, "item": "I06", "what": "시간당 요금 인상", "by": "하린", "where": "sheets:lookup!price"},
    {"id": "E08", "day": 5, "item": "I03", "what": "기준 15% -> 12%", "by": "서연", "where": "notion:Decisions/D04.기준"},
    {"id": "E09", "day": 5, "item": "I09", "what": "상태값 '대기' 추가(운영 담당이 정리 시트에 바로 적음)", "by": "하린",
     "where": "sheets:clean!status"},
    {"id": "E10", "day": 6, "item": "I12", "what": "런칭일 일주일 연기", "by": "서연", "where": "notion:Decisions/D09.날짜"},
    {"id": "E11", "day": 6, "item": "I07", "what": "예약 화면 프레임 다시 그림", "by": "민지", "where": "figma:S03_예약"},
    {"id": "E12", "day": 7, "item": "I11", "what": "주간 취소율 값 갱신", "by": "하린", "where": "sheets:metrics!B2값"},
]

# 이 주에 함께 일어나는 일(자동화를 깨는 일). 모두 공식 문서에 적힌 동작에서 골랐다.
WEEK_INCIDENTS = [
    {"day": 2, "kind": "script_edit", "note": "E04 지점은 스크립트가 붙였다. Apps Script 문서: 스크립트 실행과 API 요청은 트리거를 실행하지 않는다."},
    {"day": 3, "kind": "chained_automation", "note": "A2가 만든 할 일은 '할 일 생성 -> 담당자 알림' 자동화를 실행하지 않는다(Notion 도움말)."},
    {"day": 4, "kind": "property_rename", "note": "누군가 Notion 속성 '기준'을 '규정 기준'으로 이름을 바꿈. 이름으로 찾는 스크립트는 깨지고 id로 찾으면 그대로(Notion API 문서)."},
    {"day": 6, "kind": "creator_leaves", "note": "A1·A3을 만든 사람이 다른 프로젝트로 옮겨 이 파일 권한이 빠짐. 설치형 트리거는 만든 사람 계정으로 돈다(문서). 실패 요약 메일은 만든 사람에게 간다고 읽었다(해석). 권한이 빠지면 트리거가 실패한다는 것은 모형의 가정."},
]

# 권한 표(도구별 단계 이름은 각 도움말 기준, 확인일 2026-10-08).
# Figma: Can view / Can edit, Notion: Full access / Can edit / Can edit content / Can comment / Can view,
# Google Sheets: Viewer / Commenter / Editor (+ Owner)
PERMISSIONS = {
    "민지": {"figma": "can_edit", "notion": "can_comment", "sheets": "viewer"},    # 디자인
    "서연": {"figma": "can_view", "notion": "full_access", "sheets": "commenter"},  # 기획(결정·스펙 원본)
    "준호": {"figma": "can_view", "notion": "can_edit", "sheets": "viewer"},       # 개발
    "하린": {"figma": "can_view", "notion": "can_comment", "sheets": "editor"},     # 운영(숫자 원본)
    "가져오기 스크립트": {"figma": None, "notion": None, "sheets": "editor"},
}
# Figma의 Can view도 댓글은 달 수 있다(도움말). 그래서 '보기' 권한이어도 원본 담당에게 요청을 남길 길이 있다.
CAN_EDIT = {"can_edit", "full_access", "editor"}


def owner_of(src):
    """원본 위치의 담당. 권한 표와 짝을 이룬다."""
    tool = src.split(":")[0]
    if tool == "figma":
        return "민지"
    if tool == "notion":
        return "서연" if ("Specs" in src or "Decisions" in src) else "준호"
    return "하린"


def can_edit(person, location):
    """권한 표에서 person이 location을 고칠 수 있나. 정리 시트(clean)는 사람이 못 고친다는 5장 규칙도 함께 본다."""
    tool = location.split(":")[0]
    level = PERMISSIONS.get(person, {}).get(tool)
    if level not in CAN_EDIT:
        return False
    # 정리 시트(clean)는 수식과 가져오기 스크립트만 쓴다(5장). 사람은 원본(lookup·branches)을 고친다.
    if location.startswith("sheets:clean"):
        return person == "가져오기 스크립트"
    return True


def locations(item_id, items=None):
    """항목이 놓인 모든 위치. 원본이 맨 앞."""
    it = (ITEMS if items is None else items)[item_id]
    out = [it["src"]]
    for a, b, _ in it["edges"]:
        for x in (a, b):
            if x not in out:
                out.append(x)
    return out


def all_edges(items=None):
    """(항목, 보내는 곳, 받는 곳, 방식) 목록."""
    items = ITEMS if items is None else items
    return [(k, a, b, m) for k, it in items.items() for a, b, m in it["edges"]]


def count_methods(items=None):
    out = {"ref": 0, "human": 0, "auto": 0}
    for _, _, _, m in all_edges(items):
        out[m] += 1
    return out


# 첫 화면 손계산: 취소 규정 문구 하나가 네 곳에 있다.
HAND = {
    "places": ["Notion 스펙", "Figma 상세 화면", "Figma 완료 화면", "운영 시트 안내"],
    "miss": 0.1,       # 한 곳을 손으로 옮길 때 놓칠 확률(가정)
    "changes": 12,     # 한 주의 변경 수
    "copy_human": 3,   # 복사 방식: 원본 말고 세 곳을 손으로 고친다
    "rule_human": 1,   # 규칙 방식: 컴포넌트 하나만 손으로, 두 화면은 인스턴스, 시트는 링크
}
