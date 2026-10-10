"""1인 개발 앱 출시 실전 3장: 행 수준 보안(RLS)을 sqlite와 파이썬으로 다시 만든 모형.

data/policies.sql 의 정책을 같은 뜻의 파이썬 함수로 옮겼다. Postgres 문서의 규칙 세 가지를 따른다.
- RLS를 켰는데 맞는 정책이 하나도 없으면 "default deny": 아무 행도 보이지 않고 고칠 수 없다.
- 같은 명령에 걸린 허용(permissive) 정책은 OR로 묶는다.
- USING은 이미 있는 행 중 보이는(고칠 수 있는) 행을, WITH CHECK는 새로 생길 행이 지켜야 할 조건을 정한다.
  UPDATE 정책에 WITH CHECK가 없으면 USING을 그 자리에 쓴다.
보이지 않는 행을 고치거나 지우려 하면 오류 없이 0행이 바뀐다. WITH CHECK를 어기면 거부한다.

모드
- "off": RLS를 켜지 않은 표. 공개 키만 있으면 누구나 모든 행을 읽고 쓴다.
- "on_no_policy": RLS만 켜고 정책을 만들지 않은 표.
- "on_policies": RLS와 정책 다섯 개.

sqlite는 행을 저장하는 데만 쓴다. 같은 정책을 SQL WHERE 절로도 적어 두고(SQL_SELECT_WHERE),
파이썬 판정과 sqlite 판정이 같은 행을 고르는지 대조한다.
"""

import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
POLICIES_SQL = HERE / "data" / "policies.sql"

USERS = {"u-alice": "alice", "u-bob": "bob"}

SEED_CARDS = [
    # (id, owner_id, title, is_public)
    (1, "u-alice", "가을 신메뉴 안내", 1),
    (2, "u-alice", "임시 저장: 할인 문구 초안", 0),
    (3, "u-alice", "고객 명단 메모(가상)", 0),
    (4, "u-bob", "동아리 모집 공고 초안", 0),
    (5, "u-bob", "발표 자료 표지", 0),
]


def connect():
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("create table cards (id integer primary key, owner_id text not null, "
               "title text not null, is_public integer not null default 0)")
    db.executemany("insert into cards values (?, ?, ?, ?)", SEED_CARDS)
    db.commit()
    return db


# ---------- 정책: data/policies.sql 과 같은 뜻 ----------
# role: "anon"(로그인 안 함) 또는 "authenticated". uid: 로그인 토큰에서 서버가 꺼낸 사용자 id(없으면 None)

POLICIES = [
    {"name": "누구나 공개 카드를 읽는다", "cmd": "select", "to": {"anon", "authenticated"},
     "using": lambda row, uid: row["is_public"] == 1, "check": None},
    {"name": "자기 카드를 읽는다", "cmd": "select", "to": {"authenticated"},
     "using": lambda row, uid: uid is not None and row["owner_id"] == uid, "check": None},
    {"name": "자기 카드만 만든다", "cmd": "insert", "to": {"authenticated"},
     "using": None, "check": lambda row, uid: uid is not None and row["owner_id"] == uid},
    {"name": "자기 카드만 고친다", "cmd": "update", "to": {"authenticated"},
     "using": lambda row, uid: uid is not None and row["owner_id"] == uid,
     "check": lambda row, uid: uid is not None and row["owner_id"] == uid},
    {"name": "자기 카드만 지운다", "cmd": "delete", "to": {"authenticated"},
     "using": lambda row, uid: uid is not None and row["owner_id"] == uid, "check": None},
]

# 같은 select 정책을 SQL로 적은 것(대조용). :uid 자리에 사용자 id(없으면 NULL)
SQL_SELECT_WHERE = "(is_public = 1) OR (:is_auth = 1 AND owner_id = :uid)"


class Forbidden(Exception):
    """WITH CHECK 를 어긴 쓰기."""


class CardsTable:
    def __init__(self, mode="off"):
        assert mode in ("off", "on_no_policy", "on_policies")
        self.mode = mode
        self.db = connect()

    def _policies(self, cmd, role):
        if self.mode == "on_no_policy":
            return []
        return [p for p in POLICIES if p["cmd"] == cmd and role in p["to"]]

    def _using_ok(self, cmd, role, row, uid):
        if self.mode == "off":
            return True
        ps = self._policies(cmd, role)
        return any(p["using"](row, uid) for p in ps)  # 정책이 없으면 False(default deny)

    def _check_ok(self, cmd, role, row, uid):
        if self.mode == "off":
            return True
        ps = self._policies(cmd, role)
        return any((p["check"] or p["using"])(row, uid) for p in ps)

    def all_rows(self):
        return [dict(r) for r in self.db.execute("select * from cards order by id")]

    def select(self, role, uid):
        return [r for r in self.all_rows() if self._using_ok("select", role, r, uid)]

    def select_sql(self, role, uid):
        """같은 규칙을 sqlite WHERE 절로 고른 결과(대조용)."""
        if self.mode == "off":
            return self.all_rows()
        if self.mode == "on_no_policy":
            return []
        q = f"select * from cards where {SQL_SELECT_WHERE} order by id"
        return [dict(r) for r in self.db.execute(q, {"uid": uid, "is_auth": int(role == "authenticated")})]

    def insert(self, role, uid, row):
        if not self._check_ok("insert", role, row, uid):
            raise Forbidden("new row violates row-level security policy")
        cur = self.db.execute("insert into cards (owner_id, title, is_public) values (?, ?, ?)",
                              (row["owner_id"], row["title"], int(row.get("is_public", 0))))
        self.db.commit()
        return cur.lastrowid

    def update(self, role, uid, card_id, changes):
        n = 0
        for r in self.all_rows():
            if r["id"] != card_id or not self._using_ok("update", role, r, uid):
                continue  # 보이지 않는 행은 조용히 건너뛴다(0행)
            new = {**r, **changes}
            if not self._check_ok("update", role, new, uid):
                raise Forbidden("new row violates row-level security policy")
            self.db.execute("update cards set owner_id=?, title=?, is_public=? where id=?",
                            (new["owner_id"], new["title"], int(new["is_public"]), card_id))
            n += 1
        self.db.commit()
        return n

    def delete(self, role, uid, card_id):
        n = 0
        for r in self.all_rows():
            if r["id"] == card_id and self._using_ok("delete", role, r, uid):
                self.db.execute("delete from cards where id=?", (card_id,))
                n += 1
        self.db.commit()
        return n


def policies_in_sql_file():
    """policies.sql 에 적힌 정책 이름과 명령이 파이썬 정책과 같은지 확인할 때 쓴다."""
    import re
    text = POLICIES_SQL.read_text(encoding="utf-8")
    return re.findall(r'create policy "([^"]+)"\s+on public\.cards for (\w+)', text)
