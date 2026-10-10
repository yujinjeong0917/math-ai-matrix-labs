"""6장 모형: 1주 변경 이벤트 12개를 다섯 가지 방식으로 세 도구에 넣고, 끝났을 때 다섯 가지를 센다.

방식(regime)
- none          규칙 없음: 링크·인스턴스·임베드 없이 위치마다 따로 적는다. 바뀐 곳에서 나머지를 손으로 고친다.
- registry      등록부만: 원본 하나 + 참조(링크·인스턴스·임베드·수식). 권한·버전·변경 기록 규칙은 없다.
- rules         등록부 + 권한 표 + 버전·변경 기록 규칙.
- auto_naive    rules + 자동화 3개를 공식 문서에 적힌 함정을 피하지 않고 만든 경우.
- auto_guarded  rules + 같은 자동화 3개에 대비를 넣은 경우(시간 트리거, id로 찾기, 한 자동화 안에서 알림,
                마지막 실행 시각 점검 칸).

세는 것(설계도 6장의 다섯 관찰값)
1. 불일치: 위치 값이 항목의 최신 값과 다른 칸 수(등록부의 모든 위치 기준)
2. 체크리스트 미충족: 그중 디자인 -> 개발 넘기기 체크리스트(3장)가 보는 곳(Figma 화면, Notion 할 일)
3. checks FALSE: Sheets 점검 칸(5장 C7, C12 + 자동화 마지막 실행 점검 C13)
4. 추적 불가 변경: 넘겨준 뒤(21일째) 누가·언제·왜 바꿨는지 기록으로 답하지 못하는 변경
5. 쓴 시간(분): 아래 TIME 가정으로 센 값

모든 확률·시간은 가정이다. q(사람 확인 간선 하나를 제대로 옮길 확률)는 0.7~0.95로 바꿔 가며 잰다.
"""

import random

import productivity_ch06_graph as G
import productivity_ch06_registry as R

REGIMES = ["none", "registry", "rules", "auto_naive", "auto_guarded"]

# 시간 가정(분). 실제 측정이 아니다.
TIME = {
    "find_copies": 5,   # 규칙 없음: 이 값이 또 어디 적혀 있었는지 찾기
    "copy": 3,          # 한 곳을 손으로 고치기
    "log": 1,           # 변경 기록 한 줄
    "comment": 2,       # 권한이 없어 원본 담당에게 댓글로 요청
    "setup": 30,        # 자동화 하나 만들기(한 번)
    "fix_break": 20,    # 깨진 자동화 고치기(알아챈 뒤)
}
P_NOTE = 0.3            # 규칙이 없을 때 누군가 "왜"를 댓글로 남겨 둘 확률(가정)
REVIEW_DAY = 21         # 2주 뒤 운영 담당에게 넘길 때 기록을 다시 본다
HISTORY_DAYS = {"notion": 7, "figma": 30, "sheets": None}  # 무료 요금제 기록 보존(도움말). None = 제한 없음
AUTOMATION_CREATOR = "준호"   # A1, A3을 Apps Script로 만든 사람
CREATOR_LEAVES_DAY = 6
RENAME_DAY = 4                # Notion 속성 '기준' 이름이 바뀌는 날

CHECKLIST = lambda loc: loc.startswith("figma:") or loc.startswith("notion:Tasks")  # noqa: E731
SHEET_CHECKS = {
    "C7 목록 밖 지점 0": ("I04", "sheets:clean!lookup"),
    "C12 상태 표준값 일치": ("I09", "sheets:clean!status"),
}


def automation_alive(name, day, regime):
    """그날 자동화 name이 값을 옮기나."""
    if regime not in ("auto_naive", "auto_guarded"):
        return False
    if name in ("A1", "A3") and day >= CREATOR_LEAVES_DAY:
        return False  # 설치형 트리거는 만든 사람 계정으로 돈다. 그 사람 권한이 빠지면 실패한다.
    if name == "A3" and regime == "auto_naive" and day >= RENAME_DAY:
        return False  # 속성을 이름으로 찾는 스크립트. id로 찾으면 이름이 바뀌어도 그대로다.
    return True


AUTO_EDGE = {("I03", "sheets:metrics!threshold"): "A3", ("I08", "notion:Tasks/T17-T18"): "A2"}


def run_week(regime, q, rng, events=None, items=None):
    items = R.ITEMS if items is None else items
    events = R.EVENTS if events is None else events
    state = {(k, loc): 0 for k in items for loc in R.locations(k, items)}
    truth = {k: 0 for k in items}
    minutes = 0
    silent = []   # 오류 없이 조용히 실패한 자동화 동작
    log = []      # 사건별 기록(추적 판정용)
    blocked = 0

    def try_edge(k, a, b, m, day):
        nonlocal minutes
        if m == "ref":
            return True
        if m == "auto" and regime in ("auto_naive", "auto_guarded"):
            name = AUTO_EDGE[(k, b)]
            ok = automation_alive(name, day, regime)
            if not ok:
                silent.append((day, name, "값을 옮기지 못함"))
            return ok
        minutes += TIME["copy"]
        return rng.random() < q

    def propagate(k, start, day):
        out = {}
        for a, b, m in items[k]["edges"]:
            out.setdefault(a, []).append((b, m))
        stack = [start]
        while stack:
            u = stack.pop()
            for b, m in out.get(u, []):
                if try_edge(k, u, b, m, day):
                    state[(k, b)] = truth[k]
                    stack.append(b)

    for e in events:
        k, loc, day, who = e["item"], e["where"], e["day"], e["by"]
        truth[k] += 1
        src = items[k]["src"]
        rec = {"id": e["id"], "day": day, "tool_edited": loc.split(":")[0], "via_script": e.get("via_script", False),
               "logged": False, "auto_logged": False}
        if regime == "none":
            state[(k, loc)] = truth[k]
            minutes += TIME["find_copies"]
            for other in R.locations(k, items):
                if other != loc:
                    minutes += TIME["copy"]
                    if rng.random() < q:
                        state[(k, other)] = truth[k]
        else:
            start = loc
            if loc != src and regime != "registry" and not R.can_edit(who, loc):
                blocked += 1
                minutes += TIME["comment"] + TIME["copy"]
                start = src
                rec["tool_edited"] = src.split(":")[0]
            state[(k, start)] = truth[k]
            propagate(k, start, day)
            if start != src:  # 참조처에서 고침(권한이 막지 못함): 원본으로 거꾸로 옮기는 것도 사람 몫
                minutes += TIME["copy"]
                if rng.random() < q:
                    state[(k, src)] = truth[k]
                    propagate(k, src, day)
            if regime in ("rules", "auto_naive", "auto_guarded"):
                minutes += TIME["log"]
                rec["logged"] = rng.random() < q
                if e["id"] in ("E03", "E11"):  # 버전 규칙: 지표 정의 변경 직전, 넘기기 직전에 버전 이름 붙이기
                    rec["named_version"] = rng.random() < q
        if regime in ("auto_naive", "auto_guarded") and k == "I08":
            if regime == "auto_naive":
                # A2가 만든 할 일은 '할 일 생성 -> 담당자 알림' 자동화를 실행하지 않는다(Notion 도움말)
                silent.append((day, "A2", "만든 할 일의 담당자 알림이 안 감"))
            rec["auto_logged"] = True
        log.append(rec)

    # ---- 끝났을 때 세기(7일째 저녁)
    wrong = [(k, loc) for (k, loc), v in state.items() if v != truth[k]]
    checklist = [w for w in wrong if CHECKLIST(w[1])]
    checks = {name: state[(k, loc)] == truth[k] for name, (k, loc) in SHEET_CHECKS.items()}
    # 규칙이 없으면 lookup이 branches를 가리키지 않는 손 사본이다. 같은 기준으로 점검 칸을 계산만 해 본다
    # (실제로는 그 점검 칸을 만든 사람도 없다).
    a_dead_unknown = 0
    if regime == "auto_guarded":
        # C13: 자동화마다 '마지막 성공 실행이 24시간 안'인지 보는 칸. 지표 시트 맨 위 칸에 묶어 사람이 매일 본다.
        for name in ("A1", "A3"):
            checks[f"C13 {name} 마지막 실행 24시간 안"] = automation_alive(name, 7, regime)
    if regime == "auto_naive":
        # 실패 메일은 만든 사람에게 간다. 그 사람이 빠졌으니 아무도 모른다.
        a_dead_unknown = sum(not automation_alive(n, 7, regime) for n in ("A1", "A3"))
    checks_false = sum(not v for v in checks.values())
    hidden = [w for w in wrong if not CHECKLIST(w[1]) and w not in
              [(k, loc) for (k, loc) in SHEET_CHECKS.values()]]

    untrace = 0
    for rec in log:
        if rec["logged"] or rec["auto_logged"] or rec.get("named_version"):
            continue
        keep = HISTORY_DAYS[rec["tool_edited"]]
        hist_ok = keep is None or REVIEW_DAY - rec["day"] <= keep
        if rec["tool_edited"] == "sheets" and rec["via_script"]:
            hist_ok = False  # 셀 편집 기록에는 추가·삭제한 행이 안 보인다(도움말). 사람이 아닌 스크립트가 붙였다.
        if not (hist_ok and rng.random() < P_NOTE):
            untrace += 1

    setup = TIME["setup"] * 3 if regime.startswith("auto") else 0
    if regime == "auto_guarded":
        minutes += TIME["fix_break"] * sum(not v for k_, v in checks.items() if k_.startswith("C13"))
    return {
        "mismatch": len(wrong), "checklist_unmet": len(checklist), "checks_false": checks_false,
        "hidden_mismatch": len(hidden), "untraceable": untrace, "minutes": minutes, "setup_minutes": setup,
        "blocked_by_permission": blocked, "silent_automation_failures": len(silent),
        "dead_automation_unknown": a_dead_unknown, "wrong": wrong,
        "checks": checks, "log": log,
    }


def summarize(regime, q, n_seeds=1000, seed0=0):
    keys = ["mismatch", "checklist_unmet", "checks_false", "hidden_mismatch", "untraceable", "minutes",
            "setup_minutes", "blocked_by_permission", "silent_automation_failures", "dead_automation_unknown"]
    acc = {k: 0.0 for k in keys}
    any_wrong = 0
    for s in range(n_seeds):
        r = run_week(regime, q, random.Random(seed0 + s))
        for k in keys:
            acc[k] += r[k]
        any_wrong += r["mismatch"] > 0
    out = {k: round(v / n_seeds, 2) for k, v in acc.items()}
    out["p_any_mismatch"] = round(any_wrong / n_seeds, 3)
    return out


def expected_mismatch_from_source(events, q):
    """원본에서 시작한 이벤트만 모아, 끝났을 때 불일치 기댓값을 그래프 식으로 바로 계산한다(시뮬레이션 대조용).

    이벤트마다 위치 n이 맞을 확률은 q^(원본에서 n까지의 사람 확인 간선 수). 같은 항목이 두 번 바뀌면
    마지막 이벤트만 끝 상태를 정한다. 자동화 간선은 사람 확인 간선으로 센다(registry, rules 방식).
    """
    last = {}
    for e in events:
        assert e["where"] == R.ITEMS[e["item"]]["src"], "원본에서 시작한 이벤트만"
        last[e["item"]] = e
    total = 0.0
    for k in last:
        it = R.ITEMS[k]
        as_human = {**it, "edges": [(a, b, "human" if m == "auto" else m) for a, b, m in it["edges"]]}
        total += G.expected_mismatch(as_human, q)[0]
    return round(total, 6)


# ------------------------------------------------------------------ 자동화가 깨진 뒤 알아채기까지
def detection_days(kind, rng, p_notice=0.1, horizon=60):
    """자동화가 깨진 날부터 누군가 알아챈 날까지 걸린 일수.

    kind
    - "email_creator_here": 오류가 나고, 실패 메일을 받는 만든 사람이 아직 있다 -> 다음 날 안다.
    - "email_creator_gone": 오류는 나지만 메일 받을 사람이 없다(GitLab 2017 cron 메일 거부와 같은 모양).
    - "silent": 오류조차 없다(onEdit이 스크립트 편집에 안 돈다, 자동화가 만든 페이지가 다음 자동화를 안 부른다).
    - "heartbeat": 마지막 실행 시각 점검 칸을 사람이 매일 보는 지표 시트 맨 위에 둔다 -> 다음 날 안다.
    메일·점검 칸이 없으면, 사람이 하루에 p_notice 확률로 어긋난 값을 우연히 발견한다(가정).
    """
    if kind in ("email_creator_here", "heartbeat"):
        return 1
    for d in range(1, horizon + 1):
        if rng.random() < p_notice:
            return d
    return horizon


# ------------------------------------------------------------------ 양방향 동기화: 원본이 둘이 되는 순간
def sync_day(rng, delay_min, rate_per_hour, hours=8, one_way=False):
    """하루 동안 같은 값을 두 도구(Notion 칸, Sheets 칸)에서 고친다. 서로의 편집은 delay_min분 뒤 상대 칸을 덮어쓴다
    (도착한 값이 그대로 들어간다. 마지막 도착이 이긴다).

    반환: (편집 수, 옛 값에 덮인 편집 수, 하루가 끝났을 때 두 칸이 다른가)
    one_way=True이면 Notion만 고칠 수 있고 Sheets는 받기만 한다(원본 하나).
    """
    edits = []
    for side in ("notion", "sheets"):
        if one_way and side == "sheets":
            continue
        t = rng.expovariate(rate_per_hour / 60.0)
        while t <= hours * 60:
            edits.append((t, side))
            t += rng.expovariate(rate_per_hour / 60.0)
    edits.sort()
    writes = {"notion": [], "sheets": []}
    for i, (t, side) in enumerate(edits):  # i가 클수록 나중에 만든 편집
        other = "sheets" if side == "notion" else "notion"
        writes[side].append((t, i))
        writes[other].append((t + delay_min, i))
    clobbered = set()
    end = {}
    for side, ws in writes.items():
        ws.sort()
        cur = None
        for _, i in ws:
            if cur is not None and i < cur:
                clobbered.add(cur)  # 더 새 편집 cur가 옛 편집 i에 덮였다
            cur = i
        end[side] = cur
    diverged = len(edits) > 0 and end["notion"] != end["sheets"]
    return len(edits), len(clobbered), diverged


def two_way_sync(delay_min, rate_per_hour=1.0, days=2000, seed=0, one_way=False):
    rng = random.Random(seed)
    n_e = n_c = n_d = 0
    for _ in range(days):
        e, c, d = sync_day(rng, delay_min, rate_per_hour, one_way=one_way)
        n_e += e
        n_c += c
        n_d += d
    return {"delay_min": delay_min, "rate_per_hour": rate_per_hour, "days": days, "edits": n_e,
            "clobbered_edits": n_c, "clobbered_rate": round(n_c / max(1, n_e), 4),
            "days_diverged": n_d, "diverged_day_rate": round(n_d / days, 4)}


# ------------------------------------------------------------------ 웹훅 순서
def webhook_order(gap_sec, max_delay_sec=300, n=20000, seed=0):
    """같은 페이지를 gap_sec초 간격으로 두 번 고친다. 각 이벤트는 0~max_delay_sec초 사이 무작위로 늦게 도착한다
    (Notion 문서: 5분 안에 도착, 순서는 바뀔 수 있음).

    - 도착 순서대로 적용: 늦게 만든 편집이 먼저 도착하면 최종 값이 옛 값이다.
    - 이벤트를 신호로만 쓰고 API로 최신 값을 다시 읽기(문서 권장): 최종 값은 늘 최신.
    반환: 옛 값으로 끝난 비율(시뮬레이션)과 식 (D - g)^2 / (2 D^2)의 값.
    """
    rng = random.Random(seed)
    D = max_delay_sec
    stale = 0
    for _ in range(n):
        d1, d2 = rng.uniform(0, D), rng.uniform(0, D)
        if gap_sec + d2 < d1:  # 두 번째 편집이 먼저 도착
            stale += 1
    theory = (D - gap_sec) ** 2 / (2 * D * D) if gap_sec < D else 0.0
    return {"gap_sec": gap_sec, "stale_rate_arrival_order": round(stale / n, 4), "theory": round(theory, 4),
            "stale_rate_refetch": 0.0}
