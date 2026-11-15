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
# 확장자별 Cache-Control. HTML만 no-cache(저장하되 쓸 때마다 서버에 변경 여부 확인), 나머지는 하루(86400초) 동안 확인 없이 재사용.
CACHE_RULES = {
    ".html": "no-cache",
    ".css": "public, max-age=86400",
    ".js": "public, max-age=86400",
    ".map": "public, max-age=86400",
    ".svg": "public, max-age=86400",
    ".woff2": "public, max-age=86400",
    ".json": "public, max-age=86400",
}
# API 응답은 사용자별 결과라 저장하지 않는다.
API_CACHE = "no-store"


def load_server_config():
    return json.loads((SERVER_ONLY / "ai_config.json").read_text(encoding="utf-8"))


def generate_background_svg(user_prompt, template_id):
    """서버 쪽 '이미지 생성' 흉내. 키와 지시문은 계산에만 쓰고 응답에 넣지 않는다."""
    cfg = load_server_config()
    rules = (SERVER_ONLY / "prompt_rules.txt").read_text(encoding="utf-8")
    # 키는 가짜 문자열이며, 값이 있어야 "생성"이 진행되는 흉내만 낸다.
    assert cfg["image_api_key"]  # 키가 있어야 "생성"한다. 실제 외부 호출은 없다.
    # 입력(프롬프트, 템플릿, 서버 지시문)이 같으면 같은 그림이 나오게 해시로 무늬를 정한다. 지시문(rules)이 해시에 섞이지만 응답에는 나가지 않는다.
    seed = hashlib.sha256((user_prompt + "|" + template_id + "|" + rules).encode()).hexdigest()
    # 첫 바이트(0~255)를 색상각 0~359 범위로 환산.
    hue = int(seed[:2], 16) * 360 // 256
    shapes = []
    for k in range(6):
        # 해시의 다음 구간을 읽어 원의 좌표(0~319)를 만든다. viewBox가 320x320이라 나머지 연산 기준도 320.
        x = int(seed[2 + 4 * k: 4 + 4 * k], 16) % 320
        y = int(seed[4 + 4 * k: 6 + 4 * k], 16) % 320
        shapes.append(f'<circle cx="{x}" cy="{y}" r="{20 + k * 6}" fill="hsl({hue},60%,{70 + k * 3}%)"/>')
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="320" height="320">'
            + f'<rect width="320" height="320" fill="hsl({hue},50%,92%)"/>' + "".join(shapes) + "</svg>")


class EditorHandler(SimpleHTTPRequestHandler):
    serve_maps = True
    # 클래스 변수라 모든 연결이 공유한다. make_server가 0으로 초기화한다.
    usage = None  # 서버 쪽 사용량(사용자 구분 없이 전체 횟수만 센다)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    # 기본 구현은 요청마다 stderr에 한 줄 출력한다.
    def log_message(self, *args):  # 테스트 출력이 지저분해지지 않게
        pass

    def list_directory(self, path):  # 폴더 목록은 보여 주지 않는다
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
        return None

    def send_head(self):
        # 쿼리스트링을 뗀 경로로 .map 여부를 본다. serve_maps=False면 지도 파일만 404.
        if self.path.split("?")[0].endswith(".map") and not self.serve_maps:
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")
            return None
        return super().send_head()

    def end_headers(self):
        path = self.path.split("?")[0]
        # 오류 응답(상태 400 이상)에는 캐시 헤더를 붙이지 않는다. _status는 아래 send_response가 기록한 값.
        if not path.startswith("/api/") and getattr(self, "_status", 200) < 400:
            # "/"로 끝나는 경로는 index.html이 내려가므로 .html 규칙을 적용.
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
        # 사용자 구분 없이 서버 전체 횟수만 센다(모형의 단순화). 실제 서비스는 사용자별로 세야 한도 의미가 있다.
        EditorHandler.usage = (EditorHandler.usage or 0) + 1
        # 한도 초과는 429 Too Many Requests. 키를 쓰는 비용을 서버가 막는 지점.
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
    # serve_maps 값을 담은 하위 클래스를 만들어 서버마다 설정이 섞이지 않게 한다.
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
