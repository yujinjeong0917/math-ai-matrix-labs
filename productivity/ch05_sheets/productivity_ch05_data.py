"""5장 합성 데이터: 동네 스터디룸 예약 서비스의 2주 운영 기록.

실존 회사·인물 없음. 고정 시드로 만들고, 표준 라이브러리만 쓴다.

- 지점 20곳(B01~B20). B01~B15는 런칭 때 연 지점, B16~B20은 둘째 주에 연 지점이다.
  시트에 지점을 추가하면 아래에 붙는다(append). 이 순서가 범위 누락 예제의 무대다.
- 예약 행: booking_id, date(ISO 글자), branch, room, hours, status, customer
- 상태는 표준 네 값: 예약 / 취소 / 노쇼 / 완료

무엇을 만드나: 2주(14일) 예약 기록 행 목록과 지점별 취소 확률, 사람이 손으로 옮겨 적은 듯한 지저분한 사본(messy_copy), 행 수 상한 예제용 이벤트 로그.
입력: 시드와 하루 예약 건수 범위. 출력: 행 사전 목록(날짜는 ISO 글자, 이용 시간은 글자 "1"~"4"). 같은 시드면 항상 같은 데이터다.
이 값들은 모두 합성이다. 지점별 취소율은 DECISION_THRESHOLD(15%)와 비교하는 의사결정 예제에 쓰인다.
"""

import random
from datetime import date, timedelta

START = date(2026, 9, 21)  # 런칭 첫날(월)
DAYS = 14
STATUSES = ["예약", "취소", "노쇼", "완료"]
BRANCHES = [f"B{i:02d}" for i in range(1, 21)]
FIRST_WAVE = BRANCHES[:15]
SECOND_WAVE = BRANCHES[15:]
SECOND_WAVE_OPEN_DAY = 7  # 둘째 주 월요일에 연다
DECISION_THRESHOLD = 0.15  # 평균 취소율이 15% 이상이면 취소 규정을 바꾼다(4장 D04와 이어짐). 가상의 의사결정 기준

# 첫 화면 손계산 표: 지점 5곳(가~마), (예약 건수, 취소 건수). 앞 3곳의 평균 10% < 기준 15% <= 5곳 전체 평균 17%라 범위가 3곳에서 멈추면 결론이 달라진다
# (지점 이름은 한글 가나다 순서이고 B01.. 코드와 무관하다)
HAND = [("가", 20, 2), ("나", 25, 3), ("다", 25, 2), ("라", 20, 6), ("마", 20, 5)]


def rooms_of(branch):
    """지점마다 방 4개. 방 번호는 '층-번호' 꼴이라 자동 형식 변환에 걸리기 쉽다(예: 2-03)."""
    k = int(branch[1:])  # "B07" -> 7
    floor = 1 + k % 3
    return [f"{floor}-{n:02d}" for n in range(1, 5)]


def branch_cancel_prob(branch, rng, wave_gap=True):
    """지점별 취소 확률. wave_gap=True이면 새 지점이 조금 더 높다(낯선 지점이라 예약을 쉽게 바꾼다는 가정)."""
    base = rng.uniform(0.06, 0.16)  # 지점별 기본 취소 확률 6~16%
    if wave_gap and branch in SECOND_WAVE:
        base += rng.uniform(0.06, 0.14)
    return base


def make_bookings(seed=5, per_day=(6, 14), wave_gap=True):
    """2주 예약 기록을 만든다. 반환: (rows, cancel_prob)."""
    # 난수 소비 순서(지점별 확률 -> 노쇼율 -> 날짜별 예약)가 데이터를 결정한다. 순서를 바꾸면 같은 시드로도 다른 데이터가 나온다
    rng = random.Random(seed)
    probs = {b: branch_cancel_prob(b, rng, wave_gap) for b in BRANCHES}
    noshow = {b: rng.uniform(0.02, 0.06) for b in BRANCHES}
    customers = [f"C{i:04d}" for i in range(1, 601)]
    rows, bid = [], 1
    for d in range(DAYS):
        day = START + timedelta(days=d)
        for b in BRANCHES:
            if b in SECOND_WAVE and d < SECOND_WAVE_OPEN_DAY:  # 둘째 주 월요일(d = 7) 전에는 새 지점의 예약이 없다
                continue
            for _ in range(rng.randint(*per_day)):
                u = rng.random()  # 한 번의 난수로 취소 -> 노쇼 -> 그 외 순서로 구간을 나눈다
                if u < probs[b]:
                    st = "취소"
                elif u < probs[b] + noshow[b]:
                    st = "노쇼"
                elif d >= DAYS - 1:
                    st = "예약"  # 마지막 날은 아직 이용 전인 예약이 남는다
                else:
                    st = "완료"
                rows.append({
                    "booking_id": f"R{bid:05d}",
                    "date": day.isoformat(),
                    "branch": b,
                    "room": rng.choice(rooms_of(b)),
                    "hours": str(rng.choice([1, 2, 2, 3, 4])),
                    "status": st,
                    "customer": rng.choice(customers),
                })
                bid += 1
    return rows, probs


def branch_table(rows):
    """지점 시트: 지점, 예약 건수, 취소 건수, 취소율. 지점이 추가된 순서(B01..B20)대로 붙는다."""
    out = []
    for b in BRANCHES:
        mine = [r for r in rows if r["branch"] == b]
        n = len(mine)
        c = sum(r["status"] == "취소" for r in mine)
        # 예약이 0건인 지점은 취소율을 None으로 둔다(0으로 나누지 않기 위함, 이 값이 AVERAGE에서 빈칸처럼 취급되는 근거)
        out.append({"branch": b, "bookings": n, "cancels": c, "rate": c / n if n else None})
    return out


# ------------------------------------------------------------------ 지저분한 입력
DATE_VARIANTS = ("iso", "md", "day")  # 2026-10-05 / 10/5 / 5일 (messy_copy는 첫 값 iso를 빼고 고른다)
# 키: 표준 상태, 값: 사람이 적을 만한 표기들(첫 원소가 표준). "취소 "는 끝에 공백이 붙은 오타다
STATUS_VARIANTS = {
    "취소": ["취소", "취소됨", "cancel", "취소 "],
    "완료": ["완료", "이용완료", "done"],
    "노쇼": ["노쇼", "no-show"],
    "예약": ["예약", "예약됨"],
}


def messy_copy(rows, seed=55, p_date=0.3, p_status=0.3):
    """사람이 손으로 옮겨 적은 듯한 사본. 날짜 표기와 상태 표기를 섞는다."""
    rng = random.Random(seed)
    out = []
    for r in rows:
        r = dict(r)
        if rng.random() < p_date:
            y, m, d = (int(x) for x in r["date"].split("-"))
            kind = rng.choice(DATE_VARIANTS[1:])  # "md"는 연도를 잃고 "day"는 달까지 잃는다(정리 단계에서 후자는 복구 불가)
            r["date"] = f"{m}/{d}" if kind == "md" else f"{d}일"
        if rng.random() < p_status:
            r["status"] = rng.choice(STATUS_VARIANTS[r["status"]][1:])  # 표준 표기를 빼고 변형 하나로 바꾼다
        out.append(r)
    return out


def event_log(n_bookings, seed=7, events_per_booking=3):
    """행 수 상한 예제용 이벤트 로그. 예약 하나가 이벤트 3줄(생성·결제·결과)을 만든다.

    시각 순서대로 쌓이므로, 뒤에서 잘리면 가장 최근 날짜가 사라진다.
    """
    rng = random.Random(seed)
    days = 30  # 한 달치
    per_day = n_bookings / days
    out = []
    for i in range(n_bookings):
        d = min(int(i / per_day), days - 1)  # 예약 번호 i가 몇째 날인지. 시각순으로 쌓이므로 번호가 클수록 최근 날짜
        day = (date(2026, 9, 1) + timedelta(days=d)).isoformat()
        b = rng.choice(BRANCHES)
        for e in ["생성", "결제", rng.choice(["완료", "취소", "노쇼"])][:events_per_booking]:
            out.append((f"R{i + 1:06d}", day, b, e))
    return out
