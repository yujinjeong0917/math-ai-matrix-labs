"""1인 개발 앱 출시 실전 1장: 우리 편집기를 열 때 브라우저가 받는 요청을 흉내 낸 모형.

브라우저가 아니에요. 브라우저가 따라가는 것과 같은 참조만 따라가요.
- HTML의 <link href>, <script src>, <img src>
- CSS의 url(...)
- JS 안의 fetch 대상 문자열(템플릿 목록)과, 목록 파일이 가리키는 첫 템플릿
- 사용자가 하는 두 동작(템플릿 바꾸기, AI 배경 만들기 버튼)
- 개발자 도구에서 JavaScript source maps 설정을 켜고 열었을 때의 지도 파일

이 모형은 이 장의 실습 서버(127.0.0.1)에만 요청해요. 다른 주소는 받지 않아요.
실제 브라우저와 비교하려면 서버를 띄우고 개발자 도구 Network 탭(Disable cache 켬)의 목록과 맞춰 보세요.
"""

import contextlib
import gzip
import hashlib
import json
import re
import threading
import urllib.error
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import webapp_ch01_build as B
import webapp_ch01_server as S

ALLOWED_HOST = "127.0.0.1"


@contextlib.contextmanager
def running_editor(serve_maps=True):
    srv = S.make_server(0, serve_maps=serve_maps)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://{ALLOWED_HOST}:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def gzip_size(data):
    """압축해서 보낸다면 몇 바이트일까(이 서버는 실제로 압축하지 않는다). mtime=0으로 결정적."""
    return len(gzip.compress(data, compresslevel=9, mtime=0))


def request(base, path, method="GET", body=None):
    url = base + path
    host = urlparse(url).hostname
    if host != ALLOWED_HOST:
        raise ValueError("이 모형은 실습 서버(127.0.0.1)에만 요청해요")
    data = json.dumps(body, ensure_ascii=False).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            payload = r.read()
            return r.status, dict(r.headers), payload
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


class _Refs(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs = []  # (path, kind)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "link" and a.get("href"):
            self.refs.append((a["href"], "css" if a.get("rel") == "stylesheet" else "img"))
        elif tag == "script" and a.get("src"):
            self.refs.append((a["src"], "js"))
        elif tag == "img" and a.get("src"):
            self.refs.append((a["src"], "img"))


KIND_BY_SUFFIX = {".html": "document", ".css": "css", ".js": "js", ".svg": "img", ".woff2": "font",
                  ".json": "json", ".map": "sourcemap"}


def _norm(p):
    return "/" + p.lstrip("/")


def visit(base, actions=("change_template", "make_background"), devtools_source_maps=False):
    """요청 목록을 차례대로 돌려준다. 같은 주소는 한 번만 받는다(브라우저도 한 페이지 안에서 같은 그림을 다시 받지 않는다고 가정)."""
    log, seen = [], set()

    def get(path, initiator, step):
        path = _norm(path)
        if path in seen:
            return None
        seen.add(path)
        status, headers, payload = request(base, path)
        suffix = Path(path if path != "/" else "/index.html").suffix
        log.append({
            "step": step, "path": path, "method": "GET", "kind": KIND_BY_SUFFIX.get(suffix, "other"),
            "initiator": initiator, "status": status, "bytes": len(payload), "gzip_bytes": gzip_size(payload),
            "cache_control": headers.get("Cache-Control", ""), "sha256": hashlib.sha256(payload).hexdigest(),
            "_body": payload,
        })
        return payload

    # 1) 페이지 열기
    html = get("/", "주소창", "open").decode()
    p = _Refs()
    p.feed(html)
    js_text = ""
    for ref, kind in p.refs:
        body = get(ref, "HTML", "open")
        if body is None:
            continue
        if kind == "css":
            for u in re.findall(r'url\("([^"]+)"\)', body.decode()):
                get(u, "CSS", "open")
        if kind == "js":
            js_text = body.decode()
    # JS가 처음에 부르는 것: 템플릿 목록 -> 첫 템플릿
    index_path = next(s.strip('"') for s in B.string_literals(js_text) if s.strip('"').endswith(".json"))
    index = json.loads(get(index_path, "JS fetch", "open"))
    first = json.loads(get(index["templates"][0]["file"], "JS fetch", "open"))
    get(first["background"], "템플릿", "open")
    # 2) 사용자 동작
    if "change_template" in actions:
        get(index["templates"][1]["file"], "JS fetch", "change_template")
    if "make_background" in actions:
        status, headers, payload = request(base, "/api/generate-image", "POST",
                                           {"prompt": "파란 하늘과 종이비행기", "template": first["id"]})
        log.append({"step": "make_background", "path": "/api/generate-image", "method": "POST", "kind": "api",
                    "initiator": "JS fetch", "status": status, "bytes": len(payload),
                    "gzip_bytes": gzip_size(payload), "cache_control": headers.get("Cache-Control", ""),
                    "sha256": hashlib.sha256(payload).hexdigest(), "_body": payload})
    # 3) 개발자 도구에서 지도 파일 설정을 켜고 열었을 때
    if devtools_source_maps:
        m = re.search(r"//# sourceMappingURL=(\S+)", js_text)
        if m:
            seen.discard(_norm(m.group(1)))
            get(m.group(1), "개발자 도구", "devtools")
    return log


def public_view(log):
    return [{k: v for k, v in r.items() if k != "_body"} for r in log]


def all_downloaded_bytes(log):
    return b"".join(r["_body"] for r in log if r["status"] == 200)


def inventory(log):
    """editor/ 아래 모든 파일이 브라우저로 내려갔는지 표로 만든다."""
    ok = [r for r in log if r["status"] == 200]
    hashes = {r["sha256"] for r in ok}
    map_sources = []
    for r in ok:
        if r["kind"] == "sourcemap":
            map_sources += json.loads(r["_body"])["sourcesContent"]
    rows = []
    for f in sorted(S.EDITOR.rglob("*")):
        if not f.is_file() or "__pycache__" in f.parts or f.name.startswith("."):
            continue
        rel = f.relative_to(S.EDITOR).as_posix()
        data = f.read_bytes()
        if hashlib.sha256(data).hexdigest() in hashes:
            where = "내려감"
        elif data.decode("utf-8", "replace") in map_sources:
            where = "지도 파일 안에 내용이 들어 있어 내려감"
        elif rel.startswith("public/"):
            where = "공개 폴더에 있지만 이번 방문에서는 요청되지 않음"
        else:
            where = "서버에만 있음"
        rows.append({"file": rel, "bytes": len(data), "where": where})
    return rows


def cache_after_first_visit(log):
    """첫 방문 뒤 사용자 컴퓨터(브라우저 캐시)에 남는 것과, 하루 안에 다시 열 때 서버에 묻는 것."""
    stored, revalidate, not_stored = [], [], []
    for r in log:
        if r["status"] != 200:
            continue
        cc = r["cache_control"]
        if "no-store" in cc:
            not_stored.append(r["path"])
        elif "no-cache" in cc:
            stored.append(r["path"])
            revalidate.append(r["path"])
        elif "max-age" in cc:
            stored.append(r["path"])
    stored_bytes = sum(r["bytes"] for r in log if r["path"] in stored and r["status"] == 200)
    return {"stored": stored, "stored_bytes": stored_bytes, "ask_server_again": revalidate,
            "not_stored": not_stored}
