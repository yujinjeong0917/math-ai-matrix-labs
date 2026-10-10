"""1인 개발 앱 출시 실전 3장: 보안 다섯 가지를 켜고 끌 수 있는 작은 가상 앱(표준 라이브러리 http.server).

1장 카드뉴스 편집기에 로그인, 카드 저장(DB), 관리 화면, 배경 그림 업로드, AI 배경 만들기를 붙인 모형이다.
다섯 가지를 따로 켜고 끈다(FIXES의 키).
- rls          : cards 표의 행 수준 보안. "off" / "on_no_policy" / "on_policies"
- key_on_server: False면 브라우저 번들(app.js)에 가짜 AI 비밀 키가 들어 있다
- admin_authz  : False면 /admin 과 관리용 API가 누구에게나 열린다(링크만 화면에서 숨김)
- upload_check : False면 업로드를 그대로 받아 원래 이름으로 /uploads/ 에서 내보낸다
- rate_limit   : False면 AI 배경 만들기에 횟수 제한이 없다. True면 사용자별 토큰 버킷

127.0.0.1에만 묶는다. 바깥 네트워크, 진짜 키, 진짜 AI 호출은 없다. 세션 토큰·이메일은 모두 가짜다.

직접 띄워 보기:
    uv run python webapp/ch03_vibe_security/webapp_ch03_app.py before   # 고치기 전
    uv run python webapp/ch03_vibe_security/webapp_ch03_app.py after    # 고친 뒤
"""

import json
import mimetypes
import sys
import threading
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import webapp_ch03_ratelimit as R
import webapp_ch03_rls as L
import webapp_ch03_upload as U

HERE = Path(__file__).parent
APP = HERE / "app"
HOST = "127.0.0.1"

PUBLISHABLE_KEY = "demo_publishable_key_ch03"   # 브라우저에 넣어도 되는 공개용 키(2장)
AI_SECRET_KEY = "DEMO_SECRET_NOT_REAL_0003"      # 서버에만 둬야 하는 비밀 키(가짜)

# 서버만 아는 세션 표: 토큰 -> 사용자. 역할은 여기서만 읽는다.
SESSIONS = {
    "demo-session-alice": {"uid": "u-alice", "role": "user"},
    "demo-session-bob": {"uid": "u-bob", "role": "user"},
    "demo-session-admin": {"uid": "u-admin", "role": "admin"},
}
MEMBERS = [  # 관리 화면이 보여 주는 가입자 목록(가짜)
    {"uid": "u-alice", "email": "alice@example.com"},
    {"uid": "u-bob", "email": "bob@example.com"},
    {"uid": "u-admin", "email": "admin@example.com"},
]

BEFORE = {"rls": "off", "key_on_server": False, "admin_authz": False, "upload_check": False, "rate_limit": False}
AFTER = {"rls": "on_policies", "key_on_server": True, "admin_authz": True, "upload_check": True, "rate_limit": True}


class AppState:
    def __init__(self, fixes, clock=None):
        self.fixes = dict(fixes)
        self.clock = clock or R.FakeClock(0)
        self.cards = L.CardsTable(self.fixes["rls"])
        self.public_uploads = {}   # 고치기 전: 원래 이름 -> (내용, 브라우저가 보낸 형식)
        self.private_files = {}    # 고친 뒤: 서버가 만든 이름 -> (내용, 판정한 형식). 공개 폴더 밖
        self.limiter = R.PerUserLimiter(self.clock) if self.fixes["rate_limit"] else R.NoLimit()
        self.generated = 0         # AI 배경을 실제로 만든 횟수(= 비용이 드는 호출 수)
        self.lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    state: AppState = None

    def log_message(self, *args):
        pass

    # ---------- 공통 ----------
    def _send(self, code, body=b"", ctype="application/json; charset=utf-8", headers=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _session(self):
        auth = self.headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        return SESSIONS.get(token)

    def _body(self):
        n = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(n) if n else b""

    def _route(self):
        u = urlparse(self.path)
        return u.path, parse_qs(u.query)

    # ---------- 정적 파일 ----------
    def _static(self, path):
        folder = APP / ("public_after" if self.state.fixes["key_on_server"] else "public_before")
        files = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js"}
        if path not in files:
            return False
        name = files[path]
        ctype = "text/html; charset=utf-8" if name.endswith(".html") else "text/javascript; charset=utf-8"
        self._send(200, (folder / name).read_bytes(), ctype)
        return True

    # ---------- 1. DB(REST) : RLS ----------
    def _rest(self, method, qs):
        if self.headers.get("apikey") != PUBLISHABLE_KEY:
            return self._send(401, {"error": "no api key"})
        sess = self._session()
        role, uid = ("authenticated", sess["uid"]) if sess else ("anon", None)
        t = self.state.cards
        with self.state.lock:
            try:
                if method == "GET":
                    return self._send(200, t.select(role, uid))
                if method == "POST":
                    row = json.loads(self._body() or b"{}")
                    new_id = t.insert(role, uid, row)
                    return self._send(201, {"id": new_id})
                card_id = int(qs.get("id", ["0"])[0])
                if method == "PATCH":
                    n = t.update(role, uid, card_id, json.loads(self._body() or b"{}"))
                    return self._send(200, {"updated": n})
                if method == "DELETE":
                    return self._send(200, {"deleted": t.delete(role, uid, card_id)})
            except L.Forbidden as e:
                return self._send(403, {"error": str(e)})
        return self._send(405, {"error": "method"})

    # ---------- 3. 관리 화면 ----------
    def _admin_gate(self):
        """서버 세션에서 역할을 읽는다. 브라우저가 보낸 다른 값(머리글·주소의 표시)은 보지 않는다."""
        if not self.state.fixes["admin_authz"]:
            return True
        sess = self._session()
        if sess is None:
            self._send(401, {"error": "login required"})
            return False
        if sess["role"] != "admin":
            self._send(403, {"error": "admin only"})
            return False
        return True

    def _admin(self, method, path, qs):
        if not self._admin_gate():
            return
        if method == "GET" and path == "/admin":
            return self._send(200, (APP / "admin.html").read_bytes(), "text/html; charset=utf-8")
        if method == "GET" and path == "/admin/api/users":
            return self._send(200, MEMBERS)
        if method == "POST" and path == "/admin/api/cards/delete":
            card_id = int(qs.get("id", ["0"])[0])
            with self.state.lock:
                n = self.state.cards.db.execute("delete from cards where id=?", (card_id,)).rowcount
                self.state.cards.db.commit()
            return self._send(200, {"deleted": n})
        return self._send(404, {"error": "not found"})

    # ---------- 4. 업로드 ----------
    def _upload(self, qs):
        if self._session() is None:
            return self._send(401, {"error": "login required"})
        name = qs.get("name", [""])[0]
        ctype = self.headers.get("Content-Type", "")
        declared = int(self.headers.get("Content-Length", "0"))
        if self.state.fixes["upload_check"] and declared > U.MAX_BYTES:
            # 판정은 본문을 검사하기 전에 Content-Length 머리글로 먼저 한다.
            # 남은 본문은 연결을 깔끔하게 끝내려고 읽어서 버린다(저장하지 않는다).
            self.rfile.read(declared)
            return self._send(413, {"ok": False, "reasons": ["크기 초과"]})
        data = self._body()
        if not self.state.fixes["upload_check"]:
            self.state.public_uploads[name] = (data, ctype)
            return self._send(201, {"ok": True, "url": "/uploads/" + name})
        v = U.validate(name, ctype, data)
        if not v["ok"]:
            return self._send(400, {"ok": False, "reasons": v["reasons"]})
        stored = U.stored_name(data, v["detected"])
        self.state.private_files[stored] = (data, v["detected"])
        return self._send(201, {"ok": True, "url": "/files/" + stored})

    def _serve_upload(self, path):
        if path.startswith("/uploads/") and not self.state.fixes["upload_check"]:
            name = unquote(path[len("/uploads/"):])
            if name in self.state.public_uploads:
                data, _ = self.state.public_uploads[name]
                guessed = mimetypes.guess_type(name)[0] or "application/octet-stream"
                return self._send(200, data, guessed)  # 확장자로 형식을 정한다(고치기 전)
        if path.startswith("/files/") and self.state.fixes["upload_check"]:
            name = path[len("/files/"):]
            if name in self.state.private_files:
                data, detected = self.state.private_files[name]
                return self._send(200, data, detected, {"X-Content-Type-Options": "nosniff"})
        return self._send(404, {"error": "not found"})

    # ---------- 5. AI 배경(사용량 제한) ----------
    def _generate(self):
        sess = self._session()
        if sess is None:
            return self._send(401, {"error": "login required"})
        with self.state.lock:
            ok, retry = self.state.limiter.take(sess["uid"])
            if not ok:
                return self._send(429, {"error": "too many requests"}, headers={"Retry-After": str(retry)})
            self.state.generated += 1
        svg = '<svg xmlns="http://www.w3.org/2000/svg" width="8" height="8"><rect width="8" height="8" fill="#2f5d8a"/></svg>'
        return self._send(200, svg, "image/svg+xml")

    # ---------- 분기 ----------
    def _dispatch(self, method):
        path, qs = self._route()
        if method == "GET" and self._static(path):
            return
        if path == "/rest/v1/cards":
            return self._rest(method, qs)
        if path == "/admin" or path.startswith("/admin/"):
            return self._admin(method, path, qs)
        if method == "POST" and path == "/api/upload":
            return self._upload(qs)
        if method == "GET" and (path.startswith("/uploads/") or path.startswith("/files/")):
            return self._serve_upload(path)
        if method == "POST" and path == "/api/generate-image":
            return self._generate()
        if method == "GET" and path == "/api/me":
            s = self._session()
            return self._send(200 if s else 401, s or {"error": "login required"})
        return self._send(404, {"error": "not found"})

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_PATCH(self):
        self._dispatch("PATCH")

    def do_DELETE(self):
        self._dispatch("DELETE")


def make_server(fixes, clock=None, port=0):
    state = AppState(fixes, clock)
    handler = type("H", (Handler,), {"state": state})
    srv = ThreadingHTTPServer((HOST, port), handler)
    srv.state = state
    return srv


@contextmanager
def running(fixes, clock=None):
    srv = make_server(fixes, clock)
    t = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    try:
        yield f"http://{HOST}:{srv.server_address[1]}", srv.state
    finally:
        srv.shutdown()
        srv.server_close()


def request(base, path, method="GET", body=None, token=None, apikey=None, headers=None):
    """실습 서버(127.0.0.1)에만 보내는 요청. (상태 번호, 머리글, 본문 바이트)."""
    import urllib.error
    import urllib.request
    url = base + path
    if urlparse(url).hostname != HOST:
        raise ValueError("이 모형은 실습 서버(127.0.0.1)에만 요청해요")
    data = None
    h = dict(headers or {})
    if isinstance(body, (dict, list)):
        data = json.dumps(body, ensure_ascii=False).encode()
        h.setdefault("Content-Type", "application/json")
    elif isinstance(body, bytes):
        data = body
    if token:
        h["Authorization"] = "Bearer " + token
    if apikey:
        h["apikey"] = apikey
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "after"
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 8003
    fixes = BEFORE if mode == "before" else AFTER
    import time
    clock = (lambda: R.F(int(time.time())))  # 직접 띄울 때는 진짜 시계(초 단위)
    srv = make_server(fixes, clock, port)
    print(f"실습 앱({mode}): http://{HOST}:{srv.server_address[1]}/  (끝내려면 Ctrl+C)")
    print("세션 토큰(가짜): demo-session-alice, demo-session-bob, demo-session-admin")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
