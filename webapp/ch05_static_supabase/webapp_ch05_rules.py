"""1인 개발 앱 출시 실전 5장: data/policies.sql 의 규칙을 파이썬과 sqlite로 다시 만든 모형.

진짜 Postgres·Supabase가 아니에요. 문서에 적힌 규칙만 옮겼어요.
- RLS를 켠 표에 맞는 정책이 하나도 없으면 아무것도 못 해요(default deny).
  스토리지도 같아요: "By default Storage does not allow any uploads to buckets without RLS policies."
- 같은 명령의 허용 정책은 OR로 묶어요. USING은 이미 있는 줄, WITH CHECK는 새로 생길 줄의 조건이에요.
- 비밀 키(service_role)는 정책을 모두 건너뛰어요.
- 공개(public) 버킷은 파일 주소만 알면 누구나 받지만, 올리기·지우기·옮기기에는 여전히 정책이 필요해요.
- 덮어쓰기(upsert)는 INSERT에 더해 SELECT와 UPDATE 정책도 있어야 해요.
- 비공개 버킷 파일은 서명된 주소(signed URL)를 만들면 정해 둔 시간 동안 누구나 받을 수 있어요.

storage.foldername(name)은 경로의 폴더 목록이고 Postgres 배열은 1번부터 세요.
그래서 SQL의 (storage.foldername(name))[1] 은 파이썬의 foldername(name)[0] 이에요.
"""

import re
import sqlite3
from pathlib import Path

HERE = Path(__file__).parent
POLICIES_SQL = HERE / "data" / "policies.sql"

BUCKETS = {"templates": {"public": True}, "card-images": {"public": False}}
IMAGE_EXT = ("png", "jpg", "jpeg", "webp")

USERS = {"alice": "u-alice", "bob": "u-bob"}


# ---------- 문서에 적힌 도우미 함수 세 개 ----------

def foldername(name):
    """'public/subfolder/avatar.png' -> ['public', 'subfolder'] (문서 예시와 같아요)."""
    return name.split("/")[:-1]


def filename(name):
    return name.split("/")[-1]


def extension(name):
    f = filename(name)
    return f.rsplit(".", 1)[1] if "." in f else ""


# ---------- 스토리지 정책 (policies.sql 과 같은 뜻) ----------

def _own_folder(obj, uid):
    folders = foldername(obj["name"])
    return obj["bucket_id"] == "card-images" and uid is not None and len(folders) >= 1 and folders[0] == uid


STORAGE_POLICIES = [
    {"name": "그림: 내 폴더 파일을 받는다", "cmd": "select", "to": {"authenticated"},
     "using": _own_folder, "check": None},
    {"name": "그림: 내 폴더에 그림 파일만 올린다", "cmd": "insert", "to": {"authenticated"},
     "using": None, "check": lambda o, uid: _own_folder(o, uid) and extension(o["name"]) in IMAGE_EXT},
    {"name": "그림: 내 폴더 파일을 덮어쓴다", "cmd": "update", "to": {"authenticated"},
     "using": _own_folder, "check": _own_folder},
    {"name": "그림: 내 폴더 파일을 지운다", "cmd": "delete", "to": {"authenticated"},
     "using": _own_folder, "check": None},
]

# 같은 select·insert 규칙을 sqlite WHERE 절로 적은 것(대조용). sqlite에는 배열이 없어서
# 첫 폴더를 꺼내는 함수 folder1(name)과 ext(name)을 등록해 써요.
SQL_SELECT = "bucket_id = 'card-images' AND :uid IS NOT NULL AND folder1(name) = :uid"
SQL_INSERT_CHECK = SQL_SELECT + " AND ext(name) IN ('png', 'jpg', 'jpeg', 'webp')"


class Denied(Exception):
    """정책에 막힌 요청. 실제 응답 코드는 모형에서 정하지 않아요."""


def role_of(key, uid):
    if key == "secret":
        return "service_role"
    return "authenticated" if uid else "anon"


class Storage:
    def __init__(self, policies=STORAGE_POLICIES):
        self.policies = policies
        self.objects = {}  # (bucket, name) -> {"bucket_id", "name", "owner", "data"}
        self.signed = {}   # token -> (bucket, name, expires_at)

    def _ps(self, cmd, role):
        return [p for p in self.policies if p["cmd"] == cmd and role in p["to"]]

    def _using(self, cmd, role, obj, uid):
        return any(p["using"](obj, uid) for p in self._ps(cmd, role))

    def _check(self, cmd, role, obj, uid):
        return any((p["check"] or p["using"])(obj, uid) for p in self._ps(cmd, role))

    # 올리기. upsert=True면 같은 이름이 있을 때 덮어써요.
    def upload(self, key, uid, bucket, name, data, upsert=False):
        role = role_of(key, uid)
        obj = {"bucket_id": bucket, "name": name, "owner": uid, "data": data}
        old = self.objects.get((bucket, name))
        if role != "service_role":
            if not self._check("insert", role, obj, uid):
                raise Denied("insert")
            if old is not None:
                if not upsert:
                    raise Denied("exists")
                # 덮어쓰기: 있는 파일이 보이고(select), 고칠 수 있어야(update) 해요
                if not (self._using("select", role, old, uid) and self._using("update", role, old, uid)
                        and self._check("update", role, obj, uid)):
                    raise Denied("upsert")
        elif old is not None and not upsert:
            raise Denied("exists")
        self.objects[(bucket, name)] = obj
        return name

    # API로 받기(로그인 토큰 또는 비밀 키를 함께 보냄)
    def download(self, key, uid, bucket, name):
        obj = self.objects.get((bucket, name))
        role = role_of(key, uid)
        if obj is None or (role != "service_role" and not self._using("select", role, obj, uid)):
            raise Denied("select")  # 없는 것과 못 보는 것을 구분하지 않아요
        return obj["data"]

    # 공개 주소로 받기(키 없이)
    def public_url_get(self, bucket, name):
        if not BUCKETS[bucket]["public"]:
            raise Denied("not public")
        obj = self.objects.get((bucket, name))
        if obj is None:
            raise Denied("missing")
        return obj["data"]

    def delete(self, key, uid, bucket, name):
        obj = self.objects.get((bucket, name))
        role = role_of(key, uid)
        if obj is None or (role != "service_role" and not self._using("delete", role, obj, uid)):
            return 0  # 보이지 않는 파일은 조용히 0개
        del self.objects[(bucket, name)]
        return 1

    def create_signed_url(self, key, uid, bucket, name, expires_in, now):
        self.download(key, uid, bucket, name)  # 받을 수 있는 사람만 만들 수 있다고 둬요
        token = f"signed-{len(self.signed) + 1:04d}"
        self.signed[token] = (bucket, name, now + expires_in)
        return token

    def signed_get(self, token, now):
        bucket, name, exp = self.signed[token]
        if now >= exp:
            raise Denied("expired")
        return self.objects[(bucket, name)]["data"]


# ---------- sqlite 대조 ----------

def sqlite_select_names(objects, uid, where):
    db = sqlite3.connect(":memory:")
    db.create_function("folder1", 1, lambda n: (foldername(n) or [None])[0], deterministic=True)
    db.create_function("ext", 1, extension, deterministic=True)
    db.execute("create table objects (bucket_id text, name text)")
    db.executemany("insert into objects values (?, ?)", [(o["bucket_id"], o["name"]) for o in objects])
    rows = db.execute(f"select name from objects where {where} order by name", {"uid": uid}).fetchall()
    return [r[0] for r in rows]


def policies_in_sql_file():
    text = POLICIES_SQL.read_text(encoding="utf-8")
    return re.findall(r'create policy "([^"]+)"\s+on (\S+) for (\w+)', text)


# ---------- 시험 시나리오 ----------

def candidate_paths():
    """정책을 시험할 경로들. 내 폴더, 남의 폴더, 폴더 없음, 확장자 다름, 다른 버킷."""
    a, b = USERS["alice"], USERS["bob"]
    return [
        ("card-images", f"{a}/upload/cover.png"),
        ("card-images", f"{a}/upload/notes.txt"),
        ("card-images", f"{a}/ai/bg-001.png"),
        ("card-images", f"{b}/upload/poster.webp"),
        ("card-images", "cover.png"),
        ("card-images", f"shared/{a}/cover.png"),
        ("templates", f"{a}/notice.json"),
        ("templates", "notice.json"),
    ]


def run_storage_scenarios():
    """같은 요청을 정책 없음 / 정책 있음 / 비밀 키로 보내 결과를 표로 만들어요."""
    out = []
    a, b = USERS["alice"], USERS["bob"]
    for label, policies in (("정책 없음", []), ("정책 4개", STORAGE_POLICIES)):
        st = Storage(policies)
        # 미리 넣어 둔 파일(관리자가 비밀 키로 올림)
        st.upload("secret", None, "templates", "notice.json", b"{template}")
        st.upload("secret", None, "card-images", f"{b}/upload/poster.webp", b"bob-poster")
        st.upload("secret", None, "card-images", f"{a}/upload/old.png", b"alice-old")
        cases = [
            ("alice가 내 폴더에 png 올리기", lambda: st.upload("publishable", a, "card-images", f"{a}/upload/cover.png", b"v1")),
            ("alice가 내 폴더에 txt 올리기", lambda: st.upload("publishable", a, "card-images", f"{a}/upload/notes.txt", b"x")),
            ("alice가 bob 폴더에 올리기", lambda: st.upload("publishable", a, "card-images", f"{b}/upload/fake.png", b"x")),
            ("로그인 안 한 사람이 올리기", lambda: st.upload("publishable", None, "card-images", "anon/x.png", b"x")),
            ("alice가 템플릿 버킷에 올리기", lambda: st.upload("publishable", a, "templates", "notice.json", b"x", upsert=True)),
            ("alice가 내 파일 받기", lambda: st.download("publishable", a, "card-images", f"{a}/upload/old.png")),
            ("bob이 alice 파일 받기", lambda: st.download("publishable", b, "card-images", f"{a}/upload/old.png")),
            ("alice가 bob 파일 받기", lambda: st.download("publishable", a, "card-images", f"{b}/upload/poster.webp")),
            ("키 없이 템플릿 공개 주소로 받기", lambda: st.public_url_get("templates", "notice.json")),
            ("키 없이 비공개 버킷 주소로 받기", lambda: st.public_url_get("card-images", f"{b}/upload/poster.webp")),
            ("alice가 내 png 덮어쓰기(upsert)", lambda: st.upload("publishable", a, "card-images", f"{a}/upload/old.png", b"v2", upsert=True)),
            ("bob이 alice 파일 지우기", lambda: st.delete("publishable", b, "card-images", f"{a}/upload/old.png")),
            ("비밀 키로 bob 파일 받기", lambda: st.download("secret", None, "card-images", f"{b}/upload/poster.webp")),
        ]
        for name, fn in cases:
            try:
                r = fn()
                if isinstance(r, int):
                    res = "지움 1개" if r else "0개(보이지 않음)"
                else:
                    res = "허용"
            except Denied as e:
                res = f"거부({e})"
            except KeyError:
                res = "없음"
            out.append({"policies": label, "case": name, "result": res})
    return out


def run_upsert_needs_select_update():
    """INSERT 정책만 있을 때 덮어쓰기가 막히는지(문서: SELECT와 UPDATE도 필요)."""
    a = USERS["alice"]
    only_insert = [p for p in STORAGE_POLICIES if p["cmd"] == "insert"]
    res = {}
    for label, pol in (("insert만", only_insert), ("insert+select+update", [p for p in STORAGE_POLICIES if p["cmd"] != "delete"])):
        st = Storage(pol)
        st.upload("publishable", a, "card-images", f"{a}/upload/cover.png", b"v1")
        try:
            st.upload("publishable", a, "card-images", f"{a}/upload/cover.png", b"v2", upsert=True)
            res[label] = "덮어씀"
        except Denied as e:
            res[label] = f"거부({e})"
    return res


def run_signed_url():
    a = USERS["alice"]
    st = Storage()
    st.upload("secret", None, "card-images", f"{a}/ai/bg-001.png", b"png")
    tok = st.create_signed_url("publishable", a, "card-images", f"{a}/ai/bg-001.png", expires_in=600, now=0)
    out = {}
    for t in (0, 599, 600):
        try:
            st.signed_get(tok, now=t)
            out[str(t)] = "받음"
        except Denied:
            out[str(t)] = "만료"
    try:
        st.create_signed_url("publishable", USERS["bob"], "card-images", f"{a}/ai/bg-001.png", 600, 0)
        out["bob_create"] = "만듦"
    except Denied:
        out["bob_create"] = "거부"
    return out


def crosscheck_sqlite():
    """모든 시험 경로 x 세 사용자(alice, bob, 로그인 안 함)에서 파이썬과 sqlite가 같은 줄을 고르나."""
    objs = [{"bucket_id": bk, "name": n} for bk, n in candidate_paths()]
    agree = 0
    total = 0
    detail = {}
    for who, uid in (("alice", USERS["alice"]), ("bob", USERS["bob"]), ("anon", None)):
        role = "authenticated" if uid else "anon"
        st = Storage()
        py_sel = sorted(o["name"] for o in objs if st._using("select", role, o, uid))
        py_ins = sorted(o["name"] for o in objs if st._check("insert", role, o, uid))
        sq_sel = sqlite_select_names(objs, uid, SQL_SELECT)
        sq_ins = sqlite_select_names(objs, uid, SQL_INSERT_CHECK)
        for py, sq in ((py_sel, sq_sel), (py_ins, sq_ins)):
            total += 1
            agree += int(py == sq)
        detail[who] = {"select": len(py_sel), "insert_ok": len(py_ins)}
    return {"paths": len(objs), "comparisons": total, "agree": agree, "by_user": detail}


# ---------- 카드 표와 사용량 표 (policies.sql 의 ①, ②) ----------
# 카드 읽기 정책은 3장과 달리 하나로 합쳤어요: to anon, authenticated / is_public or auth.uid() = owner_id.
# 로그인하지 않으면 auth.uid()가 null이라 뒤쪽 비교는 참이 될 수 없어요.

SEED_CARDS = [
    (1, "u-alice", "가을 신메뉴 안내", 1),
    (2, "u-alice", "임시 저장: 할인 문구 초안", 0),
    (3, "u-alice", "발표 자료 표지", 0),
    (4, "u-bob", "동아리 모집 공고", 1),
    (5, "u-bob", "초안", 0),
]

CARD_SELECT = lambda row, uid: row["is_public"] == 1 or (uid is not None and row["owner_id"] == uid)  # noqa: E731
CARD_INSERT_CHECK = lambda row, uid: uid is not None and row["owner_id"] == uid  # noqa: E731
SQL_CARD_SELECT = "(is_public = 1) OR (:uid IS NOT NULL AND owner_id = :uid)"


class Tables:
    """cards(정책 2개)와 ai_usage(읽기 정책 1개, 쓰기 정책 없음)."""

    def __init__(self):
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.execute("create table cards (id integer primary key, owner_id text, title text, is_public integer)")
        self.db.executemany("insert into cards values (?, ?, ?, ?)", SEED_CARDS)
        self.db.execute("create table ai_usage (user_id text, day text, count integer)")
        self.db.executemany("insert into ai_usage values (?, ?, ?)",
                            [("u-alice", "2026-10-10", 2), ("u-bob", "2026-10-10", 1)])

    def cards(self, key, uid):
        rows = [dict(r) for r in self.db.execute("select * from cards order by id")]
        if role_of(key, uid) == "service_role":
            return rows
        return [r for r in rows if CARD_SELECT(r, uid)]

    def cards_sql(self, uid):
        q = f"select * from cards where {SQL_CARD_SELECT} order by id"
        return [dict(r) for r in self.db.execute(q, {"uid": uid})]

    def insert_card(self, key, uid, owner_id, title):
        row = {"owner_id": owner_id, "title": title, "is_public": 0}
        role = role_of(key, uid)
        if role != "service_role" and not (role == "authenticated" and CARD_INSERT_CHECK(row, uid)):
            raise Denied("cards insert")
        self.db.execute("insert into cards (owner_id, title, is_public) values (?, ?, 0)", (owner_id, title))
        return True

    def usage(self, key, uid):
        rows = [dict(r) for r in self.db.execute("select * from ai_usage order by user_id")]
        role = role_of(key, uid)
        if role == "service_role":
            return rows
        if role == "anon":
            return []  # anon에게 걸린 select 정책이 없어요
        return [r for r in rows if r["user_id"] == uid]

    def bump_usage(self, key, uid, user_id):
        """쓰기 정책이 하나도 없으니 비밀 키만 고칠 수 있어요(default deny)."""
        if role_of(key, uid) != "service_role":
            raise Denied("ai_usage write")
        self.db.execute("update ai_usage set count = count + 1 where user_id = ?", (user_id,))
        return True


def run_table_scenarios():
    t = Tables()
    a, b = USERS["alice"], USERS["bob"]
    out = {}
    for who, key, uid in (("로그인 안 함", "publishable", None), ("alice", "publishable", a),
                          ("bob", "publishable", b), ("비밀 키", "secret", None)):
        out[who] = {"cards": [r["id"] for r in t.cards(key, uid)], "usage_rows": len(t.usage(key, uid))}
    writes = {}
    for name, fn in (
        ("alice가 자기 카드 만들기", lambda: t.insert_card("publishable", a, a, "새 카드")),
        ("alice가 bob 이름으로 카드 만들기", lambda: t.insert_card("publishable", a, b, "가짜")),
        ("로그인 안 한 사람이 카드 만들기", lambda: t.insert_card("publishable", None, a, "x")),
        ("alice가 자기 사용량 늘리기", lambda: t.bump_usage("publishable", a, a)),
        ("비밀 키로 사용량 늘리기", lambda: t.bump_usage("secret", None, a)),
    ):
        try:
            fn()
            writes[name] = "허용"
        except Denied:
            writes[name] = "거부"
    t2 = Tables()
    sql_agree = all([r["id"] for r in t2.cards("publishable", uid)] == [r["id"] for r in t2.cards_sql(uid)]
                    for uid in (None, a, b))
    return {"reads": out, "writes": writes, "sql_agree": sql_agree}
