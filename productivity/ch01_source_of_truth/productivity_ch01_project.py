"""생산성 도구 1장 데이터: 관통 프로젝트 "동네 스터디룸 예약"의 값 3개, 자리 3곳, 시나리오 카드 8장, 원본 등록부 15항목.

값은 손으로 적었다. 실존 회사·인물 없음. 사람 이름과 역할은 6장(`ch06_connect`)과 같게 맞췄다
(민지 디자인, 서연 기획, 준호 개발, 하린 운영). 6장 폴더는 동시에 쓰이므로 import하지 않고 필요한 항목을 여기 새로 적었다.
"""

# 값이 적혀 있는 자리 세 곳
PLACES = ["notion", "figma", "sheets"]
PLACE_LABEL = {"notion": "기획 문서(Notion 스펙)", "figma": "디자인 시안(Figma)", "sheets": "운영 시트(Sheets)"}

# 값 세 개. 값마다 칸(field)이 있다. 운영 시간은 여는 시각과 닫는 시각 두 칸.
VALUES = ["price", "hours", "cancel"]
VALUE_LABEL = {"price": "시간당 요금", "hours": "운영 시간", "cancel": "취소 규정"}
INITIAL = {
    "price": {"won": 6000},
    "hours": {"open": 9, "close": 22},
    "cancel": {"free_before_h": 24},
}

# 시나리오 카드 8장. where = 이 카드를 받은 사람이 실제로 고친 곳(손에 익은 곳).
# 변경은 칸 단위다. 운영 시간 카드는 여는 시각이나 닫는 시각 하나만 바꾼다.
CARDS = [
    {"id": "C1", "by": "하린", "where": "sheets", "value": "price", "field": "won", "new": 7000,
     "what": "시간당 요금 6,000원 -> 7,000원"},
    {"id": "C2", "by": "서연", "where": "notion", "value": "cancel", "field": "free_before_h", "new": 12,
     "what": "무료 취소 24시간 전까지 -> 12시간 전까지"},
    {"id": "C3", "by": "민지", "where": "figma", "value": "hours", "field": "close", "new": 23,
     "what": "닫는 시각 22시 -> 23시(사장님 메시지를 받고 화면 문구에서 바로 고침)"},
    {"id": "C4", "by": "서연", "where": "notion", "value": "price", "field": "won", "new": 6500,
     "what": "개업 할인: 요금을 6,500원으로"},
    {"id": "C5", "by": "하린", "where": "sheets", "value": "hours", "field": "open", "new": 8,
     "what": "여는 시각 9시 -> 8시"},
    {"id": "C6", "by": "민지", "where": "figma", "value": "cancel", "field": "free_before_h", "new": 6,
     "what": "무료 취소 12시간 전까지 -> 6시간 전까지(기획자 요청을 메신저로 받음)"},
    {"id": "C7", "by": "하린", "where": "sheets", "value": "price", "field": "won", "new": 7000,
     "what": "개업 할인 끝: 요금 7,000원"},
    {"id": "C8", "by": "서연", "where": "notion", "value": "hours", "field": "close", "new": 24,
     "what": "닫는 시각 -> 24시"},
]

# 원본 등록부 방식에서 값마다 원본 자리와 나머지 자리를 잇는 방법(6장 I01·I05·I06과 같다).
#   "ref"  링크·임베드·수식 참조: 원본이 바뀌면 따라 바뀐다
#   "hand" 손으로 옮김: 등록부에 적힌 담당이 알림을 받고 옮긴다. 놓칠 수 있다
DECISION_RULE = {
    "screen": "figma",   # 화면 모양은 Figma
    "text": "notion",    # 결정·스펙 문장은 Notion
    "number": "sheets",  # 숫자 데이터는 Sheets
}
SOURCE = {"price": "sheets", "hours": "notion", "cancel": "notion"}
LINKS = {
    "price": {"notion": "ref", "figma": "hand"},    # Notion 스펙에는 시트를 임베드, 화면 글자는 손으로
    "hours": {"figma": "hand", "sheets": "hand"},   # 화면 글자와 운영 시트의 수용 계산 칸은 손으로
    "cancel": {"figma": "hand", "sheets": "ref"},   # 운영 시트에는 문구 대신 Notion 링크
}

# 첫 화면 손계산
HAND = {"q": 0.9, "k": 3, "m": 8}


def render(value, v):
    """사람이 읽는 꼴로."""
    if value == "price":
        return f"{v['won']:,}원"
    if value == "hours":
        return f"{v['open']:02d}:00-{v['close']:02d}:00"
    return f"{v['free_before_h']}시간 전까지 무료"


# 원본 등록부 15항목. 열: 항목 / 종류(screen·text·number) / 원본 도구 / 원본 자리 / 담당 / 참조하는 곳(방법) / 변경 알림
# I01~I12는 6장 규칙 표와 같은 항목·원본이고, I13~I15를 더했다.
REGISTER = [
    {"id": "I01", "name": "취소 규정 문구", "kind": "text", "tool": "notion", "where": "Specs/S02 취소 규정", "owner": "서연",
     "refs": [("figma", "Components/취소안내", "hand"), ("sheets", "metrics!안내", "ref")],
     "notify": "변경 기록 + 민지 멘션"},
    {"id": "I02", "name": "취소율 지표 정의", "kind": "text", "tool": "notion", "where": "Specs/지표정의 취소율", "owner": "서연",
     "refs": [("sheets", "metrics!B2 수식", "hand")], "notify": "변경 기록 + 하린 멘션"},
    {"id": "I03", "name": "규정 변경 기준값", "kind": "text", "tool": "notion", "where": "Decisions/D04 기준", "owner": "서연",
     "refs": [("sheets", "metrics!threshold", "hand")], "notify": "변경 기록 + 하린 멘션"},
    {"id": "I04", "name": "지점 목록", "kind": "number", "tool": "sheets", "where": "branches", "owner": "하린",
     "refs": [("sheets", "clean!lookup", "ref"), ("notion", "Projects 지점 수", "hand")], "notify": "시트 변경 알림 + 서연"},
    {"id": "I05", "name": "운영 시간", "kind": "text", "tool": "notion", "where": "Specs/S02 운영 시간", "owner": "서연",
     "refs": [("figma", "S02_상세 운영 시간", "hand"), ("sheets", "capacity!hours", "hand")], "notify": "변경 기록 + 민지·하린 멘션"},
    {"id": "I06", "name": "시간당 요금", "kind": "number", "tool": "sheets", "where": "lookup!price", "owner": "하린",
     "refs": [("figma", "S03_예약 요금", "hand"), ("notion", "Specs/S03 요금(임베드)", "ref")], "notify": "시트 변경 알림 + 민지"},
    {"id": "I07", "name": "예약 화면 프레임", "kind": "screen", "tool": "figma", "where": "S03_예약", "owner": "민지",
     "refs": [("notion", "Tasks Figma 링크", "ref")], "notify": "Figma 댓글 + 준호"},
    {"id": "I08", "name": "결정 D10(전화 인증 철회) 상태", "kind": "text", "tool": "notion", "where": "Decisions/D10 상태", "owner": "서연",
     "refs": [("notion", "Tasks T17-T18", "ref"), ("figma", "S03_예약 전화 칸", "hand")], "notify": "변경 기록 + 민지·준호 멘션"},
    {"id": "I09", "name": "예약 상태 표준값", "kind": "number", "tool": "sheets", "where": "lookup!status", "owner": "하린",
     "refs": [("sheets", "clean!status", "ref"), ("notion", "Specs/지표정의 상태 목록", "hand")], "notify": "시트 변경 알림 + 서연"},
    {"id": "I10", "name": "예약 버튼 문구", "kind": "text", "tool": "notion", "where": "Specs/S03 버튼 문구", "owner": "서연",
     "refs": [("figma", "Components/버튼", "hand")], "notify": "변경 기록 + 민지 멘션"},
    {"id": "I11", "name": "주간 취소율 값", "kind": "number", "tool": "sheets", "where": "metrics!B2", "owner": "하린",
     "refs": [("notion", "Projects 주간 리포트", "hand")], "notify": "시트 변경 알림 + 서연"},
    {"id": "I12", "name": "런칭일", "kind": "text", "tool": "notion", "where": "Decisions/D09 날짜", "owner": "서연",
     "refs": [("notion", "Tasks T19-T20", "ref"), ("sheets", "metrics!기간", "hand")], "notify": "변경 기록 + 하린 멘션"},
    {"id": "I13", "name": "방별 최대 인원", "kind": "number", "tool": "sheets", "where": "rooms!capacity", "owner": "하린",
     "refs": [("figma", "S02_상세 인원", "hand"), ("notion", "Specs/S02 인원(임베드)", "ref")], "notify": "시트 변경 알림 + 민지"},
    {"id": "I14", "name": "목록 화면 구성", "kind": "screen", "tool": "figma", "where": "S01_목록", "owner": "민지",
     "refs": [("notion", "Specs/S01 화면 링크", "ref")], "notify": "Figma 댓글 + 서연·준호"},
    {"id": "I15", "name": "문의 연락처 문구", "kind": "text", "tool": "notion", "where": "Specs/S04 문의", "owner": "서연",
     "refs": [("figma", "S04_완료 문의", "hand"), ("sheets", "metrics!안내 연락처", "ref")], "notify": "변경 기록 + 민지 멘션"},
]
OWNERS = {"민지", "서연", "준호", "하린"}
