"""1인 개발 앱 출시 실전 1장: 실습용 카드뉴스 편집기 서버(표준 라이브러리 http.server).

- editor/public/ 폴더만 내보낸다. editor/src/, editor/server_only/는 주소로 닿지 않는다.
- 폴더 목록 보여 주기(SimpleHTTPRequestHandler의 기본 동작)는 끈다.
- POST /api/generate-image 는 서버에만 있는 설정(가짜 키)과 지시문을 써서 배경 SVG를 "만든다".
  진짜 AI를 부르지 않고, 입력으로 정해지는 무늬를 그리는 흉내다.
- 지도 파일(.map)을 내보낼지는 serve_maps로 정한다.

직접 띄워 보기(우리 편집기만 대상):
    uv run python webapp/ch01_everything_visible/webapp_ch01_server.py
    -> 브라우저에서 http://127.0.0.1:8000/ 을 열고 개발자 도구 Network 탭을 본다.
"""

import hashlib
import json
import sys
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
EDITOR = HERE / "editor"
PUBLIC = EDITOR / "public"
SERVER_ONLY = EDITOR / "server_only"

# 파일 종류별 캐시 규칙(실습에서 정한 값)
CACHE_RULES = {
    ".html": "no-cache",
    ".css": "public, max-age=86400",
    ".js": "public, max-age=86400",
    ".map": "public, max-age=86400",
    ".svg": "public, max-age=86400",
    ".woff2": "public, max-age=86400",
    ".json": "public, max-age=86400",
}
API_CACHE = "no-store"


def load_server_config():
    return json.loads((SERVER_ONLY / "ai_config.json").read_text(encoding="utf-8"))


def generate_background_svg(user_prompt, template_id):
    """서버 쪽 '이미지 생성' 흉내. 키와 지시문은 계산에만 쓰고 응답에 넣지 않는다."""
    cfg = load_server_config()
    rules = (SERVER_ONLY / "prompt_rules.txt").read_text(encoding="utf-8")
    assert cfg["image_api_key"]  # 키가 있어야 "생성"한다. 실제 외부 호출은 없다.
    seed = hashlib.sha256((user_prompt + "|" + template_id + "|" + rules).encode()).hexdigest()
    hue = int(seed[:2], 16) * 360 // 256
    shapes = []
    for k in range(6):
        x = int(seed[2 + 4 * k: 4 + 4 * k], 16) % 320
        y = int(seed[4 + 4 * k: 6 + 4 * k], 16) % 320
        shapes.append(f'<circle cx="{x}" cy="{y}" r="{20 + k * 6}" fill="hsl({hue},60%,{70 + k * 3}%)"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="320" height="320">'
            + f'<rect width="320" height="320" fill="hsl({hue},50%,92%)"/>' + "".join(shapes) + "</svg>")


class EditorHandler(SimpleHTTPRequestHandler):
    serve_maps = True
    usage = None  # 서버 쪽 사용량(사용자 구분 없이 전체 횟수만 센다)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    def log_message(self, *args):  # 테스트 출력이 지저분해지지 않게
        pass

    def list_directory(self, path):  # 폴더 목록은 보여 주지 않는다
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
        return None

    def send_head(self):
        if self.path.split("?")[0].endswith(".map") and not self.serve_maps:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return None
        return super().send_head()

    def end_headers(self):
        path = self.path.split("?")[0]
        if not path.startswith("/api/") and getattr(self, "_status", 200) < 400:
            suffix = Path(path if not path.endswith("/") else path + "index.html").suffix
            rule = CACHE_RULES.get(suffix)
            if rule:
                self.send_header("Cache-Control", rule)
        super().end_headers()

    def send_response(self, code, message=None):
        self._status = code
        super().send_response(code, message)

    def do_POST(self):
        if self.path != "/api/generate-image":
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(length) or b"{}")
        cfg = load_server_config()
        EditorHandler.usage = (EditorHandler.usage or 0) + 1
        if EditorHandler.usage > cfg["daily_limit_per_user"]:
            self.send_error(HTTPStatus.TOO_MANY_REQUESTS, "Too Many Requests")
            return
        body = generate_background_svg(str(req.get("prompt", "")), str(req.get("template", ""))).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/svg+xml; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", API_CACHE)
        self.end_headers()
        self.wfile.write(body)


def make_server(port=0, serve_maps=True):
    """127.0.0.1에만 묶인 서버를 만든다(같은 컴퓨터에서만 접속)."""
    handler = type("Handler", (EditorHandler,), {"serve_maps": serve_maps})
    EditorHandler.usage = 0
    return ThreadingHTTPServer(("127.0.0.1", port), partial(handler))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    srv = make_server(port)
    print(f"실습용 편집기: http://127.0.0.1:{srv.server_address[1]}/  (끝내려면 Ctrl+C)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
