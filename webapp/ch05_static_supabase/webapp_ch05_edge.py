"""1인 개발 앱 출시 실전 5장: AI 배경 만들기를 Edge Function으로 옮긴 모형.

진짜 Supabase Edge Function(Deno)이 아니에요. supabase/functions/generate-background/index.ts 와
같은 순서를 파이썬으로 따라 해요.

플랫폼 단계(모형)
  OPTIONS(사전 확인 요청)는 처리하고 요금에 세지 않아요. 문서: "Preflight (OPTIONS) requests are not billed."
  verify_jwt = true(기본값)이면 로그인 토큰이 맞는지 먼저 보고, 틀리면 함수 코드에 닿기 전에 401.
함수 단계
  1. 토큰에서 사용자 id(sub)를 꺼내요.
  2. 설명 글 길이를 확인해요(200자 넘으면 400).
  3. ai_usage 표에서 오늘 사용량을 비밀 키 권한으로 읽고, 하루 한도(가상 5회)를 넘으면 429.
  4. 비밀 키 AI_IMAGE_KEY(환경 변수)로 AI를 불러 그림을 받아요. 실습은 가짜 AI가 작은 PNG를 만들어요.
  5. 그림을 card-images/<내 id>/ai/ 에 비밀 키 권한으로 올리고, 10분짜리 서명된 주소를 돌려줘요.

로그인 토큰은 HMAC-SHA256으로 서명한 JWT 모양의 모형이에요. Supabase가 실제로 쓰는 서명 방식과
같다고 주장하지 않아요. 비밀 값은 모두 DEMO_ 로 시작하는 가짜예요.
"""

import base64
import hashlib
import hmac
import json
import sqlite3
import struct
import zlib

import webapp_ch05_rules as R

JWT_SECRET = b"DEMO_JWT_SIGNING_SECRET_0005"   # 모형 서명용 가짜 값
ENV = {"AI_IMAGE_KEY": "DEMO_AI_IMAGE_SECRET_0005"}  # Edge Function 비밀(가짜)
DAILY_LIMIT = 5                               # 사용자당 하루 AI 배경 횟수(가상)
MAX_PROMPT = 200
SIGNED_SECONDS = 600


def _b64(b):
    return base64.urlsafe_b64encode(b).decode().rstrip("=")


def _unb64(s):
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(sub, exp, role="authenticated", secret=JWT_SECRET):
    head = _b64(json.dumps({"alg": "HS256", "typ": "JWT"}, separators=(",", ":")).encode())
    body = _b64(json.dumps({"sub": sub, "role": role, "exp": exp}, separators=(",", ":")).encode())
    sig = _b64(hmac.new(secret, f"{head}.{body}".encode(), hashlib.sha256).digest())
    return f"{head}.{body}.{sig}"


def verify_token(token, now):
    """맞으면 claims, 아니면 None. 서명 → 만료 순서로 봐요."""
    try:
        head, body, sig = token.split(".")
    except ValueError:
        return None
    want = _b64(hmac.new(JWT_SECRET, f"{head}.{body}".encode(), hashlib.sha256).digest())
    if not hmac.compare_digest(want, sig):
        return None
    claims = json.loads(_unb64(body))
    if claims.get("exp", 0) <= now:
        return None
    return claims


def fake_ai_png(prompt, key):
    """가짜 AI. 키가 맞아야 그려요. 설명 글로 정해지는 색의 8x8 PNG(표준 라이브러리 zlib)."""
    if key != ENV["AI_IMAGE_KEY"]:
        raise PermissionError("AI key rejected")
    h = hashlib.sha256(prompt.encode()).digest()
    rgb = bytes(h[:3])
    raw = b"".join(b"\x00" + rgb * 8 for _ in range(8))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


class Platform:
    """Edge Function 하나와 그 주변(사용량 표, 스토리지, 호출 수 세기)."""

    def __init__(self, verify_jwt=True):
        self.verify_jwt = verify_jwt
        self.storage = R.Storage()
        self.db = sqlite3.connect(":memory:")
        self.db.execute("create table ai_usage (user_id text, day text, count integer, primary key (user_id, day))")
        self.billed = 0          # 함수 코드까지 닿은 호출(응답 코드와 상관없이 셈)
        self.rejected_by_platform = 0  # verify_jwt 단계에서 막힌 호출. 요금에 드는지는 [확인 필요]
        self.preflight = 0
        self.log = []

    def usage(self, uid, day):
        r = self.db.execute("select count from ai_usage where user_id=? and day=?", (uid, day)).fetchone()
        return r[0] if r else 0

    def request(self, method, headers, body, now, day):
        if method == "OPTIONS":
            self.preflight += 1
            return self._out(204, {}, "사전 확인")
        auth = headers.get("Authorization", "")
        token = auth[7:] if auth.startswith("Bearer ") else ""
        claims = verify_token(token, now) if token else None
        if self.verify_jwt and claims is None:
            self.rejected_by_platform += 1
            return self._out(401, {"error": "invalid or missing JWT"}, "플랫폼이 막음")
        self.billed += 1
        return self.handler(claims, body, now, day)

    def handler(self, claims, body, now, day):
        if claims is None or claims.get("role") != "authenticated":
            return self._out(401, {"error": "login required"}, "함수가 막음")
        uid = claims["sub"]
        prompt = str(body.get("prompt", ""))
        if not prompt or len(prompt) > MAX_PROMPT:
            return self._out(400, {"error": f"prompt must be 1..{MAX_PROMPT} chars"}, "설명 글 길이")
        used = self.usage(uid, day)
        if used >= DAILY_LIMIT:
            return self._out(429, {"error": "daily limit", "used": used, "limit": DAILY_LIMIT}, "하루 한도")
        png = fake_ai_png(prompt, ENV["AI_IMAGE_KEY"])
        self.db.execute("insert into ai_usage values (?, ?, 1) on conflict(user_id, day) do update set count = count + 1",
                        (uid, day))
        self.db.commit()
        path = f"{uid}/ai/{day}-{used + 1:03d}.png"
        self.storage.upload("secret", None, "card-images", path, png)
        tok = self.storage.create_signed_url("secret", None, "card-images", path, SIGNED_SECONDS, now)
        return self._out(200, {"path": path, "signed": tok, "expires_in": SIGNED_SECONDS, "bytes": len(png)}, "만듦")

    def _out(self, status, payload, note):
        text = json.dumps(payload, ensure_ascii=False)
        self.log.append({"status": status, "note": note, "body": text})
        return status, payload


def run_scenarios():
    """같은 하루 동안의 요청 12개를 차례로 보내요."""
    now, day = 1_000_000, "2026-10-10"
    a, b = R.USERS["alice"], R.USERS["bob"]
    good_a = make_token(a, now + 3600)
    good_b = make_token(b, now + 3600)
    expired = make_token(a, now - 1)
    forged = make_token(a, now + 3600, secret=b"DEMO_WRONG_SECRET")
    p = Platform()
    steps = [
        ("사전 확인(OPTIONS)", "OPTIONS", {}, {}),
        ("토큰 없이", "POST", {}, {"prompt": "파란 하늘"}),
        ("다른 비밀로 서명한 토큰", "POST", {"Authorization": f"Bearer {forged}"}, {"prompt": "파란 하늘"}),
        ("만료된 토큰", "POST", {"Authorization": f"Bearer {expired}"}, {"prompt": "파란 하늘"}),
        ("alice, 설명 글 201자", "POST", {"Authorization": f"Bearer {good_a}"}, {"prompt": "가" * 201}),
    ]
    steps += [(f"alice {i}번째", "POST", {"Authorization": f"Bearer {good_a}"}, {"prompt": f"종이비행기 {i}"})
              for i in range(1, 7)]
    steps += [("bob 1번째", "POST", {"Authorization": f"Bearer {good_b}"}, {"prompt": "노을"})]
    rows = []
    for name, m, h, body in steps:
        status, payload = p.request(m, h, body, now, day)
        rows.append({"step": name, "status": status, "note": p.log[-1]["note"]})
    # 결과 그림을 누가 받을 수 있나
    access = {}
    path = f"{a}/ai/{day}-001.png"
    for who, uid in (("alice", a), ("bob", b)):
        try:
            p.storage.download("publishable", uid, "card-images", path)
            access[who] = "받음"
        except R.Denied:
            access[who] = "거부"
    all_responses = "".join(x["body"] for x in p.log)
    return {
        "steps": rows,
        "status_counts": {str(s): sum(1 for r in rows if r["status"] == s) for s in sorted({r["status"] for r in rows})},
        "billed_invocations": p.billed,
        "rejected_by_platform": p.rejected_by_platform,
        "preflight": p.preflight,
        "alice_usage": p.usage(a, day), "bob_usage": p.usage(b, day),
        "ai_files": sorted(n for (bk, n) in p.storage.objects if bk == "card-images"),
        "ai_png_bytes": len(fake_ai_png("종이비행기 1", ENV["AI_IMAGE_KEY"])),
        "who_can_read_alice_ai": access,
        "key_in_any_response": ENV["AI_IMAGE_KEY"] in all_responses,
    }


def run_without_verify_jwt():
    """verify_jwt를 끈 판: 토큰 없는 호출도 함수 코드에 닿아 요금에 세져요(함수 안에서 401)."""
    p = Platform(verify_jwt=False)
    now, day = 1_000_000, "2026-10-10"
    for _ in range(3):
        p.request("POST", {}, {"prompt": "x"}, now, day)
    return {"billed_invocations": p.billed, "statuses": [x["status"] for x in p.log]}
