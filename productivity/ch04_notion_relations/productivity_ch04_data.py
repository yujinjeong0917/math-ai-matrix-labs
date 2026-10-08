"""생산성 도구 4장 데이터: 관통 프로젝트 "동네 스터디룸 예약"의 결정 10개, 할 일 20개, 스펙 4개(합성).

값은 손으로 적었다. 실존 회사·인물 없음. 날짜는 2026년 9월의 가상 일정이다.
"""

PROJECT = {"id": "P01", "name": "스터디룸 예약 런칭", "status": "진행"}

# 스펙(화면). edited = 스펙 페이지의 마지막 수정일
SPECS = {
    "S01": {"title": "목록 화면", "edited": "2026-09-10"},
    "S02": {"title": "상세 화면", "edited": "2026-09-24"},
    "S03": {"title": "예약 화면", "edited": "2026-09-29"},
    "S04": {"title": "완료 화면", "edited": "2026-09-20"},
}

# 결정 로그. specs = 결정이 직접 영향을 준 화면, supersedes = 번복한 결정
DECISIONS = {
    "D01": {"title": "예약 단위를 1시간으로", "date": "2026-09-03", "owner": "준호", "specs": ["S03"], "supersedes": None},
    "D02": {"title": "목록 기본 정렬을 거리순으로", "date": "2026-09-04", "owner": "민지", "specs": ["S01"], "supersedes": None},
    "D03": {"title": "빈 목록 문구 확정", "date": "2026-09-08", "owner": "민지", "specs": ["S01"], "supersedes": None},
    "D04": {"title": "취소 규정 변경: 24시간 전까지 무료 취소", "date": "2026-09-22", "owner": "서연",
            "specs": ["S02", "S03", "S04"], "supersedes": None},
    "D05": {"title": "결제는 현장 결제만", "date": "2026-09-10", "owner": "서연", "specs": ["S03", "S04"], "supersedes": None},
    "D06": {"title": "완료 화면에 캘린더 추가 버튼", "date": "2026-09-15", "owner": "도윤", "specs": ["S04"], "supersedes": None},
    "D07": {"title": "예약 확정 전 전화번호 인증", "date": "2026-09-12", "owner": "준호", "specs": ["S03"], "supersedes": None},
    "D08": {"title": "상세 화면 사진 최대 5장", "date": "2026-09-16", "owner": "도윤", "specs": ["S02"], "supersedes": None},
    "D09": {"title": "런칭일을 10월 둘째 주로", "date": "2026-09-18", "owner": "서연", "specs": [], "supersedes": None},
    "D10": {"title": "전화번호 인증 철회, 간편 로그인으로 대체", "date": "2026-09-28", "owner": "준호",
            "specs": ["S03"], "supersedes": "D07"},
}

# D09는 화면과 무관하다고 결정 페이지에 적어 둔 결정이다(빈칸 검사에서 제외).
NO_SPEC_OK = {"D09"}

# 할 일. decisions = 근거 결정(여러 개 가능), specs = 관련 화면, updated = 마지막 수정일
TASKS = {
    "T01": {"title": "목록 정렬 기준 바꾸기", "status": "완료", "owner": "민지", "updated": "2026-09-12", "decisions": ["D02"], "specs": ["S01"]},
    "T02": {"title": "거리 계산 API 붙이기", "status": "완료", "owner": "준호", "updated": "2026-09-11", "decisions": ["D02"], "specs": ["S01"]},
    "T03": {"title": "빈 목록 문구 넣기", "status": "완료", "owner": "민지", "updated": "2026-09-12", "decisions": ["D03"], "specs": ["S01"]},
    "T04": {"title": "사진 슬라이드 5장 제한", "status": "진행", "owner": "도윤", "updated": "2026-09-25", "decisions": ["D08"], "specs": ["S02"]},
    "T05": {"title": "시간 선택을 1시간 칸으로", "status": "완료", "owner": "준호", "updated": "2026-09-30", "decisions": ["D01"], "specs": ["S03"]},
    "T06": {"title": "요금 계산을 시간 단위로", "status": "완료", "owner": "준호", "updated": "2026-09-30", "decisions": ["D01"], "specs": []},
    "T07": {"title": "목록 불러오는 중 화면", "status": "검토", "owner": "민지", "updated": "2026-09-26", "decisions": ["D02"], "specs": ["S01"]},
    "T08": {"title": "상세 화면 취소 규정 문구", "status": "완료", "owner": "서연", "updated": "2026-09-23", "decisions": ["D04"], "specs": ["S02"]},
    "T09": {"title": "완료 화면 취소 안내", "status": "진행", "owner": "도윤", "updated": "2026-09-25", "decisions": ["D04"], "specs": ["S04"]},
    "T10": {"title": "취소 수수료 계산 바꾸기", "status": "할 일", "owner": "준호", "updated": "2026-09-22", "decisions": ["D04"], "specs": []},
    "T11": {"title": "결제 안내 문구 넣기", "status": "완료", "owner": "서연", "updated": "2026-09-14", "decisions": ["D05"], "specs": ["S03", "S04"]},
    "T12": {"title": "온라인 결제 버튼 빼기", "status": "완료", "owner": "준호", "updated": "2026-09-13", "decisions": ["D05"], "specs": ["S03"]},
    "T13": {"title": "전화번호 입력칸 만들기", "status": "완료", "owner": "민지", "updated": "2026-09-20", "decisions": ["D07", "D10"], "specs": ["S03"]},
    "T14": {"title": "취소 규정 이메일 문구", "status": "할 일", "owner": "서연", "updated": "2026-09-22", "decisions": ["D04"], "specs": []},
    "T15": {"title": "캘린더 추가 버튼", "status": "완료", "owner": "도윤", "updated": "2026-09-25", "decisions": ["D06"], "specs": ["S04"]},
    "T16": {"title": "인증 문자 발송 연동", "status": "진행", "owner": "준호", "updated": "2026-09-27", "decisions": ["D07", "D10"], "specs": []},
    "T17": {"title": "간편 로그인 붙이기", "status": "할 일", "owner": "준호", "updated": "2026-09-28", "decisions": ["D10"], "specs": ["S03"]},
    "T18": {"title": "전화번호 입력칸 빼기", "status": "할 일", "owner": "민지", "updated": "2026-09-28", "decisions": ["D10"], "specs": ["S03"]},
    "T19": {"title": "전 화면 QA", "status": "할 일", "owner": "도윤", "updated": "2026-09-18", "decisions": ["D09"], "specs": ["S01", "S02", "S03", "S04"]},
    "T20": {"title": "런칭 공지 작성", "status": "할 일", "owner": "서연", "updated": "2026-09-18", "decisions": ["D09"], "specs": []},
}

DONE = "완료"


def links():
    """잃을 수 있는 연결 목록. (종류, 결정, 상대) 튜플. 순서는 고정."""
    out = []
    for d in sorted(DECISIONS):
        out.append(("owner", d, DECISIONS[d]["owner"]))
        for s in DECISIONS[d]["specs"]:
            out.append(("spec", d, s))
    for t in sorted(TASKS):
        for d in TASKS[t]["decisions"]:
            out.append(("task", d, t))
    return out
