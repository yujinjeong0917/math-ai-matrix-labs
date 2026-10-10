"""AI 도구 실전 2장: API 요청 메시지 읽기와 "작은 SDK" 흉내.

네트워크에 나가지 않는다. 실제 SDK가 대신 해 주는 일 몇 가지(키 읽기, 머리글 붙이기, 필수 칸 확인,
일시적 오류 재시도)를 표준 라이브러리로 흉내 내고, 가짜 전송 함수(FakeTransport)로 결정적으로 확인한다.

근거(확인일 2026-10-10)
- 요청 칸: https://platform.claude.com/docs/en/api/messages/create (max_tokens, messages, model은 본문 칸, system은 선택)
- 머리글: 같은 문서의 curl 예(anthropic-version: 2023-06-01, X-Api-Key)
- 재시도: https://platform.claude.com/docs/en/cli-sdks-libraries/sdks/python
  "Connection errors ..., 408 Request Timeout, 409 Conflict, 429 Rate Limit, and >=500 Internal errors are all retried by default."
  기본 2번, "short exponential backoff". 대기 시간의 정확한 값은 문서에 없어 여기서는 0.5초 × 2^k로 정했다(가정).
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
REQUIRED = ("model", "max_tokens", "messages")
RETRY_STATUS = {408, 409, 429}


def load_example(path=HERE / "data" / "request_example.json"):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def walk_blocks(req):
    """요청을 캐시 순서(system -> messages)대로 블록 목록으로 편다. (위치, 역할, 캐시 지점 여부, 글 앞부분)"""
    out = []
    sys_part = req.get("system", [])
    if isinstance(sys_part, str):
        sys_part = [{"type": "text", "text": sys_part}]
    for b in sys_part:
        out.append(("system", "system", "cache_control" in b, b["text"][:20]))
    for i, msg in enumerate(req["messages"]):
        content = msg["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        for b in content:
            out.append((f"messages[{i}]", msg["role"], "cache_control" in b, b["text"][:20]))
    return out


def breakpoints_of(req):
    return [i for i, b in enumerate(walk_blocks(req)) if b[2]]


def validate(params):
    missing = [k for k in REQUIRED if k not in params]
    if missing:
        raise ValueError(f"필수 칸이 빠졌어요: {', '.join(missing)}")
    if not isinstance(params["max_tokens"], int) or params["max_tokens"] < 0:
        raise ValueError("max_tokens는 0 이상의 정수여야 해요")


class APIError(Exception):
    def __init__(self, status, attempts):
        super().__init__(f"HTTP {status} after {attempts} attempts")
        self.status = status
        self.attempts = attempts


class MiniClient:
    """SDK가 대신 해 주는 일을 눈에 보이게 줄인 흉내."""

    def __init__(self, transport, env=None, api_key=None, max_retries=2, sleep=None):
        env = env or {}
        self.api_key = api_key or env.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("API 키가 없어요. ANTHROPIC_API_KEY 환경 변수를 확인하세요")
        self.transport = transport
        self.max_retries = max_retries
        self.sleep = sleep or (lambda s: None)
        self.waits = []

    def headers(self):
        return {"x-api-key": self.api_key, "anthropic-version": API_VERSION,
                "content-type": "application/json"}

    def create(self, **params):
        validate(params)
        body = json.dumps(params, ensure_ascii=False)
        attempt = 0
        while True:
            attempt += 1
            status, data = self.transport(URL, self.headers(), body)
            if status == 200:
                return {"attempts": attempt, **data}
            retryable = status in RETRY_STATUS or status >= 500
            if not retryable or attempt > self.max_retries:
                raise APIError(status, attempt)
            wait = 0.5 * 2 ** (attempt - 1)
            self.waits.append(wait)
            self.sleep(wait)


class FakeTransport:
    """정해 둔 상태 코드를 차례로 돌려주는 가짜 전송. 마지막에는 200과 usage를 돌려준다."""

    def __init__(self, statuses, usage=None):
        self.statuses = list(statuses)
        self.usage = usage or {"input_tokens": 0, "cache_creation_input_tokens": 0,
                               "cache_read_input_tokens": 0, "output_tokens": 0}
        self.calls = []

    def __call__(self, url, headers, body):
        self.calls.append({"url": url, "headers": dict(headers), "body": json.loads(body)})
        status = self.statuses.pop(0) if self.statuses else 200
        if status == 200:
            return 200, {"type": "message", "role": "assistant", "usage": self.usage}
        return status, {"type": "error"}
