"""8장. 교대 근무를 어떻게 짤까: 제약 프로그래밍의 생각을 작은 역추적 탐색으로 재현.

표준 라이브러리만 쓴다. 이 코드는 OR-Tools CP-SAT이 아니다. '변수 + 논리 제약 + 탐색'이라는
생각만 작은 크기로 직접 구현한 것이다. 모든 숫자는 예시용 가상값이다.

입력: 4명 x 7일, 하루 낮 2명 + 야간 1명, 연속 근무 3일 이하, 주 6일 이하, 야간 주 2회 이하,
휴무 희망 7건(어기면 벌점 1). 출력: 규칙을 지키면서 벌점이 가장 작은 일정과 방문 노드 수.
대응(README 기대 출력): 첫 가능 일정 노드 15, 벌점 5 / 벌점 최소 4, 가지치기 노드 301(가지치기 없이 421, 가능한 일정 48개) /
연속 근무 상한 4면 벌점 3 / 주 근무 상한 5, 야간 상한 1, 연속 근무 2는 불가능.
노드 = dfs 호출 횟수(가지치기로 돌아온 호출도 센다).
"""
import time
from itertools import combinations

DAY, NIGHT, OFF = "D", "N", "O"
PEOPLE = ("민", "서", "준", "하")
DAYS = 7
NEED_DAY, NEED_NIGHT = 2, 1          # 하루에 필요한 낮 근무자 수, 야간 근무자 수
MAX_CONSEC = 3                         # 연속 근무일 상한
MAX_WORK = 6                           # 주간 근무일 상한
MAX_NIGHT = 2                          # 주간 야간 상한
# 휴무 희망(사람, 요일 0~6). 어기면 벌점 1.
REQUESTS = {("민", 2), ("서", 2), ("준", 2), ("하", 4), ("민", 5), ("서", 5), ("하", 6)}


def violations(sched, people=PEOPLE, days=DAYS, need_day=NEED_DAY, need_night=NEED_NIGHT,
               max_consec=MAX_CONSEC, max_work=MAX_WORK, max_night=MAX_NIGHT):
    """완성된 일정이 어긴 하드 제약을 글로 돌려준다(없으면 빈 목록). 탐색과 독립된 검사기."""
    bad = []
    for d in range(days):
        # 인원 수는 "정확히" 같아야 한다(이상이 아님): 하루 낮 2명, 야간 1명 고정
        if sum(1 for p in people if sched[p][d] == DAY) != need_day:
            bad.append(f"{d}일차 낮 인원")
        if sum(1 for p in people if sched[p][d] == NIGHT) != need_night:
            bad.append(f"{d}일차 야간 인원")
    for p in people:
        row = sched[p]
        for d in range(days - 1):
            # 야간 다음 날은 휴무. 마지막 날 야간은 다음 날이 일정 밖이라 검사하지 않는다
            if row[d] == NIGHT and row[d + 1] != OFF:
                bad.append(f"{p} 야간 다음 날 휴식 위반({d}일차)")
        run = 0
        for x in row:
            # run = 현재까지 연속 근무일(낮·야간 모두 근무로 센다), 휴무에서 0으로 초기화
            run = run + 1 if x != OFF else 0
            if run > max_consec:
                bad.append(f"{p} 연속 근무 초과")
                break
        if sum(1 for x in row if x != OFF) > max_work:
            bad.append(f"{p} 주간 근무일 초과")
        if sum(1 for x in row if x == NIGHT) > max_night:
            bad.append(f"{p} 야간 횟수 초과")
    return bad


# 벌점은 소프트 제약: 휴무 희망일에 근무가 잡히면 1점(하드 제약과 달리 어겨도 일정은 성립)
def penalty(sched, requests=REQUESTS, days=DAYS, people=PEOPLE):
    return sum(1 for (p, d) in requests if p in people and d < days and sched[p][d] != OFF)


def solve(people=PEOPLE, days=DAYS, requests=REQUESTS, need_day=NEED_DAY, need_night=NEED_NIGHT,
          max_consec=MAX_CONSEC, max_work=MAX_WORK, max_night=MAX_NIGHT, optimize=True, prune=True):
    """하루씩 '그날의 모든 사람 배정'을 정하는 역추적 탐색.

    optimize=True면 벌점이 가장 작은 일정을 찾는다(분기한정).
    prune=False면 가지치기 없이 전체를 훑는다(비교용, 아주 작은 크기에서만).
    반환: dict(sched, penalty, nodes, solutions). 가지치기를 켜면 solutions는 끝까지 도달한 일정 수(개선될 때만 도달)이고,
    prune=False일 때는 하드 제약을 지키는 모든 일정의 수다.
    """
    n = len(people)
    # 그날 한 줄(사람별 D/N/O)의 후보를 조합으로 미리 전부 만든다: C(4,2) x C(2,1) = 12개
    # 사람 번호 조합으로 그날의 열(낮 need_day명, 야간 need_night명)을 미리 만든다.
    columns = []
    for dset in combinations(range(n), need_day):
        rest = [i for i in range(n) if i not in dset]
        for nset in combinations(rest, need_night):
            col = [OFF] * n
            for i in dset:
                col[i] = DAY
            for i in nset:
                col[i] = NIGHT
            columns.append(tuple(col))
    # 날짜별로 피할 수 없는 벌점의 아래 한계: 일하는 사람은 정해져 있어 휴무는 n - need 명뿐이다.
    # free = 근무가 없는 사람 수(4-2-1 = 1). 그날 휴무 희망자가 free보다 많으면 그 초과분만큼은 어쩔 수 없이 벌점
    free = n - need_day - need_night
    floor_pen = []
    for d in range(days):
        want_off = sum(1 for p in people if (p, d) in requests)
        floor_pen.append(max(0, want_off - free))
    # suffix[d] = d일차 이후 피할 수 없는 벌점의 합(남은 날의 하한). 가지치기의 낙관적 추정치
    suffix = [0] * (days + 1)
    for d in range(days - 1, -1, -1):
        suffix[d] = suffix[d + 1] + floor_pen[d]

    best = {"pen": None, "cols": None}
    stats = {"nodes": 0, "solutions": 0}
    cols = []

    def dfs(d, last, run, work, nights, pen):
        stats["nodes"] += 1
        # 지금 벌점 + 앞으로 최소 벌점이 이미 찾은 최선 이상이면 더 볼 필요 없다(같은 값도 버려 개선이 있을 때만 갱신)
        if prune and optimize and best["pen"] is not None and pen + suffix[d] >= best["pen"]:
            return
        if d == days:
            stats["solutions"] += 1
            if best["pen"] is None or pen < best["pen"]:
                best["pen"], best["cols"] = pen, list(cols)
            return
        for col in columns:
            # 하드 제약을 사람별로 확인: 야간 다음 날 휴무, 연속 근무 상한, 주간 근무 상한, 야간 횟수 상한
            ok = True
            for i in range(n):
                x = col[i]
                if last[i] == NIGHT and x != OFF:
                    ok = False
                    break
                if x != OFF:
                    if run[i] + 1 > max_consec or work[i] + 1 > max_work:
                        ok = False
                        break
                    if x == NIGHT and nights[i] + 1 > max_night:
                        ok = False
                        break
            if not ok:
                continue
            # 이 열을 고르면 새로 생기는 벌점: 휴무 희망인데 근무가 잡힌 사람 수
            add = sum(1 for i in range(n) if (people[i], d) in requests and col[i] != OFF)
            cols.append(col)
            dfs(d + 1, col,
                tuple(run[i] + 1 if col[i] != OFF else 0 for i in range(n)),
                tuple(work[i] + (col[i] != OFF) for i in range(n)),
                tuple(nights[i] + (col[i] == NIGHT) for i in range(n)),
                pen + add)
            cols.pop()
            # optimize=False면 첫 가능 일정을 찾는 즉시 탐색을 접는다(README의 "첫 가능 일정")
            if not optimize and best["pen"] is not None:
                return

    zero = tuple([0] * n)
    dfs(0, tuple([OFF] * n), zero, zero, zero, 0)
    if best["cols"] is None:
        return {"sched": None, "penalty": None, "nodes": stats["nodes"], "solutions": 0}
    sched = {people[i]: [best["cols"][d][i] for d in range(days)] for i in range(n)}
    return {"sched": sched, "penalty": best["pen"], "nodes": stats["nodes"], "solutions": stats["solutions"]}


def show(sched, days=DAYS):
    print("      " + " ".join(f"{d}" for d in range(days)))
    for p, row in sched.items():
        print(f"  {p}   " + " ".join(row))


def main():
    t = time.perf_counter()
    first = solve(optimize=False)
    t1 = time.perf_counter() - t
    print("첫 번째 가능한 일정: 노드", first["nodes"], "벌점", first["penalty"], "violations", violations(first["sched"]))
    show(first["sched"])
    t = time.perf_counter()
    opt = solve(optimize=True)
    t2 = time.perf_counter() - t
    print("벌점 최소 일정: 노드", opt["nodes"], "벌점", opt["penalty"], "violations", violations(opt["sched"]), "(가지치기 탐색이 끝까지 간 일정 수 %d: 5점 해, 4점 해)" % opt["solutions"])
    show(opt["sched"])
    print("휴무 희망 7건 중 못 지킨 것:", [(p, d) for (p, d) in sorted(REQUESTS, key=lambda x: (x[1], x[0])) if opt["sched"][p][d] != OFF])
    # 시간은 기기마다 달라 본문 숫자로 쓰지 않는다
    print("(실행 시간은 컴퓨터마다 달라서 본문에 쓰지 않는다)", round(t1, 3), round(t2, 3))
    print("제약을 하나씩 바꿔 보기:")
    for name, kw in [("기준", {}), ("연속 근무 상한 4 (벌점 하한 3까지 내려감)", {"max_consec": 4}), ("연속 근무 상한 2", {"max_consec": 2}), ("야간 상한 1", {"max_night": 1}),
                     ("주간 근무 상한 5 (불가능)", {"max_work": 5}), ("야간 상한 1 + 연속 2", {"max_night": 1, "max_consec": 2})]:
        r = solve(**kw)
        print(" ", name, "벌점", r["penalty"], "노드", r["nodes"])


if __name__ == "__main__":
    main()
