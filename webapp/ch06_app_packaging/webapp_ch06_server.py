"""1인 개발 앱 출시 실전 6장: PWA로 바꾼 카드뉴스 편집기를 내 컴퓨터에서 띄우는 서버(표준 라이브러리).

- pwa/public/ 만 내보낸다. 폴더 목록은 끈다.
- 서비스 워커는 https 이거나 localhost·127.0.0.1 에서만 등록된다. 그래서 127.0.0.1 로 띄운다.
- sw.js 는 no-cache(쓰기 전에 서버 확인)로 보낸다. 새 버전을 올렸을 때 브라우저가 바로 알아채게 하려는 실습 설정이다.
- POST /api/generate-image 는 진짜 AI 없이 정해진 SVG 하나를 돌려준다(no-store). 오프라인에서는 당연히 실패한다.

    uv run python webapp/ch06_app_packaging/webapp_ch06_server.py   # http://127.0.0.1:8000/
"""

import sys
from contextlib import contextmanager
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

HERE = Path(__file__).parent
PUBLIC = HERE / "pwa" / "public"

CACHE_RULES = {"sw.js": "no-cache", ".html": "no-cache"}
DEFAULT_CACHE = "public, max-age=86400"
STUB_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="320" height="320">'
            '<rect width="320" height="320" fill="#dbe7f3"/></svg>')


class PwaHandler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,
                      ".json": "application/json", ".js": "text/javascript", ".webmanifest": "application/manifest+json",
                      ".woff2": "font/woff2", ".svg": "image/svg+xml", ".png": "image/png"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    def log_message(self, *args):
        pass

    def list_directory(self, path):
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
        return None

    def end_headers(self):
        path = self.path.split("?")[0]
        if not path.startswith("/api/"):
            rule = next((v for k, v in CACHE_RULES.items() if path.endswith(k)), None)
            if path == "/":
                rule = CACHE_RULES[".html"]
            self.send_header("Cache-Control", rule or DEFAULT_CACHE)
        super().end_headers()

    def do_POST(self):
        if self.path != "/api/generate-image":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        body = STUB_SVG.encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/svg+xml")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
        self.wfile.write(body)


@contextmanager
def running(port=0):
    httpd = ThreadingHTTPServer(("127.0.0.1", port), PwaHandler)
    t = Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    httpd = ThreadingHTTPServer(("127.0.0.1", port), PwaHandler)
    print(f"http://127.0.0.1:{port}/  (Ctrl+C로 끄기)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
