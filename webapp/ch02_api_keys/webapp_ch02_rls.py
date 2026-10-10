"""1인 개발 앱 출시 실전 2장: Supabase의 행 수준 보안(RLS)을 sqlite로 흉내 낸 모형.

진짜 Postgres가 아니에요. sqlite에는 RLS가 없어서, Postgres가 하는 일을 순서대로 파이썬으로 따라 해요.
1. 권한(grant)이 없으면 정책을 보기도 전에 권한 오류(Postgres 오류 코드 42501)
2. 역할이 service_role(비밀 키)이면 정책을 건너뛰고 모든 행(BYPASSRLS)
3. RLS가 꺼져 있으면 모든 행
4. RLS가 켜져 있으면, 그 역할에 걸린 select 정책들의 조건을 OR로 묶어 WHERE에 붙임. 정책이 없으면 0행

정책 조건은 project/supabase/policies.sql의 두 정책과 같은 뜻으로 적었어요.
"""

import sqlite3

USERS = {"alice": "u-alice", "bob": "u-bob", "carol": "u-carol"}

# (주인, 전체 카드 수, 그중 공개 카드 수)
CARDS_PLAN = [("alice", 5, 2), ("bob", 3, 1), ("carol", 4, 0)]

# policies.sql과 같은 뜻: (정책 이름, 적용 역할, WHERE 조건)
POLICIES = [
    ("Anyone can read public cards", {"anon", "authenticated"}, "is_public = 1"),
    ("Owners can read their own cards", {"authenticated"}, "owner_id = :uid"),
]


class PermissionDenied(Exception):
    code = "42501"


def make_db():
    db = sqlite3.connect(":memory:")
    db.execute("create table cards (id integer primary key, owner_id text, title text, is_public integer)")
    n = 0
    for owner, total, public in CARDS_PLAN:
        for i in range(total):
            n += 1
            db.execute("insert into cards values (?, ?, ?, ?)",
                       (n, USERS[owner], f"{owner}의 카드 {i + 1}", 1 if i < public else 0))
    db.commit()
    return db


def role_of(key, user=None):
    """Supabase 문서의 표: publishable 키 + 로그인 안 함 = anon, + 로그인 = authenticated, secret 키 = service_role."""
    if key == "secret":
        return "service_role"
    return "authenticated" if user else "anon"


def select_cards(db, key, user=None, rls=True, policies=POLICIES, grants=("anon", "authenticated", "service_role")):
    role = role_of(key, user)
    if role not in grants:
        raise PermissionDenied(role)
    sql = "select id, owner_id, is_public from cards"
    params = {"uid": USERS.get(user)}
    if role != "service_role" and rls:
        conds = [c for _, roles, c in policies if role in roles]
        sql += " where " + (" or ".join(f"({c})" for c in conds) if conds else "0")
    return db.execute(sql + " order by id", params).fetchall()


def select_cards_python(rows, key, user=None, rls=True, policies=POLICIES):
    """같은 규칙을 SQL 없이 파이썬으로만 다시 계산해요(대조용)."""
    role = role_of(key, user)
    if role == "service_role" or not rls:
        return list(rows)
    names = {n for n, roles, _ in policies if role in roles}
    uid = USERS.get(user)
    out = []
    for r in rows:
        ok = (("Anyone can read public cards" in names and r[2] == 1)
              or ("Owners can read their own cards" in names and r[1] == uid))
        if ok:
            out.append(r)
    return out


SCENARIOS = [
    # (이름, 키, 사용자, rls, 정책, grants)
    ("RLS 꺼짐 · 공개용 키만", "publishable", None, False, POLICIES, None),
    ("RLS 켬 · 정책 없음 · 공개용 키만", "publishable", None, True, [], None),
    ("RLS 켬 · 정책 2개 · 공개용 키만", "publishable", None, True, POLICIES, None),
    ("RLS 켬 · 정책 2개 · alice 로그인", "publishable", "alice", True, POLICIES, None),
    ("RLS 켬 · 정책 2개 · bob 로그인", "publishable", "bob", True, POLICIES, None),
    ("RLS 켬 · 정책 2개 · carol 로그인", "publishable", "carol", True, POLICIES, None),
    ("RLS 켬 · 정책 2개 · 비밀 키", "secret", None, True, POLICIES, None),
    ("RLS 꺼짐 · anon 권한(grant) 없음 · 공개용 키만", "publishable", None, False, POLICIES, ("authenticated", "service_role")),
]


def run_scenarios():
    db = make_db()
    all_rows = db.execute("select id, owner_id, is_public from cards order by id").fetchall()
    out = []
    for name, key, user, rls, pol, grants in SCENARIOS:
        kw = {"rls": rls, "policies": pol}
        if grants is not None:
            kw["grants"] = grants
        rec = {"scenario": name, "key": key, "user": user, "role": role_of(key, user), "rls": rls,
               "policies": len(pol)}
        try:
            rows = select_cards(db, key, user, **kw)
            rec["rows"] = len(rows)
            rec["python_rows"] = len(select_cards_python(all_rows, key, user, rls, pol))
            rec["owners"] = {u: sum(1 for r in rows if r[1] == USERS[u]) for u in USERS}
        except PermissionDenied as e:
            rec["rows"] = None
            rec["error"] = f"permission denied ({e.code})"
        out.append(rec)
    return out
