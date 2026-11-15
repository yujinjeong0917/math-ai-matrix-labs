"""생산성 도구 4장: 관계형 방식(Notion 데이터베이스의 relation·rollup을 흉내 낸 표).

같은 질문을 두 방식으로 구현한다.
1. 순수 파이썬: 선택(sigma), 조인(join), 묶어 세기(gamma)를 반복문으로.
2. sqlite3(표준 라이브러리): 같은 표를 SQL로.
두 결과가 같은지는 experiments.py의 E6과 테스트에서 무작위 데이터로 대조한다.

DB 모양(사전):
  decisions: {D: {"title", "date", "owner"(없으면 None), "supersedes"}}
  tasks:     {T: {"title", "status", "owner", "updated"}}
  specs:     {S: {"title", "edited"}}
  task_dec:  {(T, D)}   할 일 ↔ 근거 결정 relation (양방향으로 보이는 한 쌍)
  dec_spec:  {(D, S)}   결정 ↔ 영향받는 스펙 relation
  task_spec: {(T, S)}   할 일 ↔ 관련 스펙 relation

무엇을 재는가: 연결(relation)이 데이터로 저장돼 있을 때 같은 질문 다섯 개를 필터·조인·묶어 세기로 답하고, 연결을 잃으면 빈칸 검사로 얼마나 드러나는지.
입력: build(lost)가 만든 DB 사전. 출력: 질문별 답, rollup(결정별 미완료 할 일 수), 빈칸 검사 결과. 같은 값을 SQL(sqlite3)로도 계산해 대조한다.
관계형 방식은 연결이 구조로 있어 "빈 relation"이 곧 눈에 띄는 신호가 되지만, 한 칸에 연결이 둘 이상 있을 때 하나를 잃으면 칸이 비지 않아 못 잡는다(테스트의 "두 번째 연결" 사례).
"""

import sqlite3

import productivity_ch04_data as DATA

DONE = DATA.DONE


def build(lost=frozenset()):
    """정답 데이터에서 DB를 만든다. lost에 든 연결(DATA.links()의 튜플)은 빠진다."""
    db = {
        "decisions": {}, "tasks": {}, "specs": {},
        "task_dec": set(), "dec_spec": set(), "task_spec": set(),
    }
    for d, v in DATA.DECISIONS.items():
        owner = None if ("owner", d, v["owner"]) in lost else v["owner"]
        db["decisions"][d] = {"title": v["title"], "date": v["date"], "owner": owner, "supersedes": v["supersedes"]}
        for s in v["specs"]:
            if ("spec", d, s) not in lost:
                db["dec_spec"].add((d, s))
    for t, v in DATA.TASKS.items():
        db["tasks"][t] = {k: v[k] for k in ("title", "status", "owner", "updated")}  # 연결 정보(decisions, specs)는 별도 표로 뺀다
        for d in v["decisions"]:
            if ("task", d, t) not in lost:
                db["task_dec"].add((t, d))
        for s in v["specs"]:
            db["task_spec"].add((t, s))  # 할 일-스펙 연결은 lost 대상이 아니다(data.links()에 없다)
    for s, v in DATA.SPECS.items():
        db["specs"][s] = dict(v)
    return db


# ---------------------------------------------------------------- 관계 대수 세 가지
def select(rows, pred):
    """sigma: 조건에 맞는 행만 남긴다."""
    return [r for r in rows if pred(r)]


def join(left, right, lkey, rkey):
    """자연 조인(중첩 반복). 키가 같은 두 행을 합친다. 시간은 두 행 수의 곱(작은 데이터용)이다. 겹치는 키는 오른쪽 값이 덮는다."""
    out = []
    for a in left:
        for b in right:
            if a[lkey] == b[rkey]:
                out.append({**a, **b})
    return out


def group_count(rows, key, all_keys=()):
    """gamma: key로 묶어 개수를 센다. all_keys에 있는데 행이 없으면 0.
    all_keys가 없으면 행이 0개인 결정이 결과에서 빠진다(SQL의 LEFT JOIN이 0을 살리는 것과 같은 역할)."""
    out = {k: 0 for k in all_keys}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def _task_rows(db):
    """tasks 사전을 행 목록(dict)으로 펼친다. 할 일 번호는 "task" 열."""
    return [{"task": t, **v} for t, v in db["tasks"].items()]


def _td_rows(db):
    """task_dec 쌍 집합을 행 목록으로 펼친다(열: task, decision). 집합이라 순서가 없다."""
    return [{"task": t, "decision": d} for t, d in db["task_dec"]]


# ---------------------------------------------------------------- rollup
def rollup_open_tasks(db):
    """결정별 미완료 할 일 수 = gamma_{decision;count}( sigma_{status != 완료}(Tasks) ⋈ TaskDec )."""
    open_tasks = select(_task_rows(db), lambda r: r["status"] != DONE)
    joined = join(open_tasks, _td_rows(db), "task", "task")
    return group_count(joined, "decision", sorted(db["decisions"]))


def rollup_open_tasks_formula(db):
    """같은 값을 Notion 수식 모양으로: prop("Tasks").filter(current.prop("Status") != "Done").length()."""
    out = {}
    for d in sorted(db["decisions"]):
        related = [t for t, dd in db["task_dec"] if dd == d]          # prop("Tasks")
        out[d] = len([t for t in related if db["tasks"][t]["status"] != DONE])  # .filter(...).length()
    return out


def project_rollups(db):
    """프로젝트 수준 rollup: 결정 수, 미완료 할 일 수."""
    return {
        "decisions": len(db["decisions"]),
        "open_tasks": len([t for t, v in db["tasks"].items() if v["status"] != DONE]),
    }


# ---------------------------------------------------------------- 질문 5개
def q1(db, d="D04"):
    """Q1: 결정 d를 근거로 한 할 일(정렬 목록)."""
    return sorted({t for t, dd in db["task_dec"] if dd == d})


def q2(db, s="S03"):
    """Q2: 화면 s에 직접 영향을 준 결정(결정-스펙 relation 기준)."""
    return sorted({d for d, ss in db["dec_spec"] if ss == s})


def q2_via_tasks(db, s="S03"):
    """연결 함정: 결정 → 할 일 → 스펙 경로를 따라가서 '영향을 준 결정'을 고른다.
    할 일에 스펙이 안 달렸거나 결정이 화면과 무관하게 할 일만 걸려 있으면 직접 연결(q2)과 결과가 달라진다."""
    tasks_on_s = {t for t, ss in db["task_spec"] if ss == s}
    return sorted({d for t, d in db["task_dec"] if t in tasks_on_s})


def q3(db):
    """Q3: 미완료 할 일이 하나라도 있는 결정(rollup이 0보다 큰 결정)."""
    r = rollup_open_tasks(db)
    return sorted(d for d, n in r.items() if n > 0)


def q4(db):
    """Q4: 결정별 결정자. 결정자가 비어 있으면 None이 그대로 나온다(페이지 방식은 이 경우 전체를 답할 수 없음으로 처리)."""
    return {d: v["owner"] for d, v in sorted(db["decisions"].items())}


def q5(db):
    """Q5: 완료된 할 일이 스펙의 마지막 수정일보다 먼저 수정된 스펙(할 일 끝난 뒤 스펙이 또 바뀐 것). 날짜 문자열의 사전순 비교를 쓴다."""
    stale = set()
    for t, s in db["task_spec"]:
        tv = db["tasks"][t]
        if tv["status"] == DONE and tv["updated"] < db["specs"][s]["edited"]:
            stale.add(s)
    return sorted(stale)


def answer_all(db):
    """질문 다섯 개의 답을 한 사전에."""
    return {"Q1": q1(db), "Q2": q2(db), "Q3": q3(db), "Q4": q4(db), "Q5": q5(db)}


def view_rows(db):
    """질문마다 답이 나온 보기에 보이는 행 수(읽은 양의 대리 지표)."""
    return {
        "Q1": len(q1(db)),                                   # Tasks 보기, 근거 결정 = D04 필터
        "Q2": len(q2(db)),                                   # Decisions 보기, 영향받는 스펙 ∋ S03 필터
        "Q3": len(db["decisions"]),                          # Decisions 보기 + 미완료 rollup 열
        "Q4": len(db["decisions"]),                          # Decisions 보기, 결정자 열
        "Q5": len(db["task_spec"]),                          # Tasks × 관련 스펙 보기
        # 위 행 수는 "그 보기를 열었을 때 눈으로 훑을 행"을 읽은 양으로 어림한 값이다(모형의 선택)
    }


# ---------------------------------------------------------------- 빈칸 검사
def empty_checks(db):
    """빈 relation·속성으로 드러나는 것들."""
    tasks_with_dec = {t for t, _ in db["task_dec"]}
    decs_with_spec = {d for d, _ in db["dec_spec"]}
    return {
        "task_no_decision": sorted(t for t in db["tasks"] if t not in tasks_with_dec),
        "decision_no_owner": sorted(d for d, v in db["decisions"].items() if v["owner"] is None),
        "decision_no_spec": sorted(d for d in db["decisions"] if d not in decs_with_spec and d not in DATA.NO_SPEC_OK),
    }


def detected_links(db, lost):
    """잃은 연결 가운데 빈칸 검사로 드러나는 것. 드러난 칸은 사람이 원래 연결을 모두 다시 채운다고 둔다."""
    # 연결을 잃었어도 그 결정/할 일에 다른 연결이 남아 있으면 칸이 비지 않아 드러나지 않는다(예: 근거 결정이 둘인 할 일)
    chk = empty_checks(db)
    found = set()
    for kind, d, x in lost:
        if kind == "owner" and d in chk["decision_no_owner"]:
            found.add((kind, d, x))
        elif kind == "spec" and d in chk["decision_no_spec"]:
            found.add((kind, d, x))
        elif kind == "task" and x in chk["task_no_decision"]:
            found.add((kind, d, x))
    return found


# ---------------------------------------------------------------- sqlite3 대응 구현
SCHEMA = """
CREATE TABLE decisions(id TEXT PRIMARY KEY, title TEXT, date TEXT, owner TEXT, supersedes TEXT REFERENCES decisions(id));
CREATE TABLE tasks(id TEXT PRIMARY KEY, title TEXT, status TEXT, owner TEXT, updated TEXT);
CREATE TABLE specs(id TEXT PRIMARY KEY, title TEXT, edited TEXT);
CREATE TABLE task_dec(task TEXT REFERENCES tasks(id), decision TEXT REFERENCES decisions(id), PRIMARY KEY(task, decision));
CREATE TABLE dec_spec(decision TEXT REFERENCES decisions(id), spec TEXT REFERENCES specs(id), PRIMARY KEY(decision, spec));
CREATE TABLE task_spec(task TEXT REFERENCES tasks(id), spec TEXT REFERENCES specs(id), PRIMARY KEY(task, spec));
"""


def to_sqlite(db):
    """파이썬 DB 사전을 메모리 SQLite로 옮긴다. 쌍 집합은 정렬해 넣어 삽입 순서를 고정한다."""
    con = sqlite3.connect(":memory:")
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO decisions VALUES(?,?,?,?,?)",
                    [(d, v["title"], v["date"], v["owner"], v["supersedes"]) for d, v in db["decisions"].items()])
    con.executemany("INSERT INTO tasks VALUES(?,?,?,?,?)",
                    [(t, v["title"], v["status"], v["owner"], v["updated"]) for t, v in db["tasks"].items()])
    con.executemany("INSERT INTO specs VALUES(?,?,?)", [(s, v["title"], v["edited"]) for s, v in db["specs"].items()])
    con.executemany("INSERT INTO task_dec VALUES(?,?)", sorted(db["task_dec"]))
    con.executemany("INSERT INTO dec_spec VALUES(?,?)", sorted(db["dec_spec"]))
    con.executemany("INSERT INTO task_spec VALUES(?,?)", sorted(db["task_spec"]))
    return con


def sql_answer_all(con, d1="D04", s2="S03"):
    col = lambda q, *a: [r[0] for r in con.execute(q, a)]  # noqa: E731
    # 미완료 수: LEFT JOIN 조건에 status != 완료를 넣어야 결정은 남기고 미완료 할 일만 센다(WHERE에 넣으면 0인 결정이 사라진다)
    open_by_dec = dict(con.execute(
        """SELECT d.id, COUNT(t.id) FROM decisions d
           LEFT JOIN task_dec td ON td.decision = d.id
           LEFT JOIN tasks t ON t.id = td.task AND t.status != ?
           GROUP BY d.id ORDER BY d.id""", (DONE,)).fetchall())
    return {
        "Q1": col("SELECT DISTINCT task FROM task_dec WHERE decision = ? ORDER BY task", d1),
        "Q2": col("SELECT DISTINCT decision FROM dec_spec WHERE spec = ? ORDER BY decision", s2),
        "Q3": sorted(d for d, n in open_by_dec.items() if n > 0),
        "Q4": dict(con.execute("SELECT id, owner FROM decisions ORDER BY id").fetchall()),
        "Q5": col("""SELECT DISTINCT s.id FROM task_spec ts JOIN tasks t ON t.id = ts.task JOIN specs s ON s.id = ts.spec
                     WHERE t.status = ? AND t.updated < s.edited ORDER BY s.id""", DONE),
        "rollup_open": open_by_dec,
        "q2_via_tasks": col("""SELECT DISTINCT td.decision FROM task_spec ts JOIN task_dec td ON td.task = ts.task
                               WHERE ts.spec = ? ORDER BY td.decision""", s2),
    }
