"""1인 개발 앱 출시 실전 3장: 카드 배경 그림 업로드 검증.

OWASP File Upload Cheat Sheet의 점검 항목 중 표준 라이브러리로 할 수 있는 것만 넣었다.
1) 크기 제한  2) 파일 이름 규칙(길이·글자)  3) 확장자 허용 목록(마지막 확장자, 대소문자 무시)
4) 브라우저가 보낸 Content-Type이 허용 목록에 있고 확장자와 맞는지(참고용, 믿지는 않는다)
5) 파일 앞부분 바이트(매직 바이트)가 확장자가 말하는 형식과 맞는지
통과한 파일은 서버가 만든 이름으로 공개 폴더 밖에 두고, 돌려줄 때는 서버가 판정한 형식만 Content-Type으로 쓴다.

매직 바이트는 WHATWG MIME Sniffing 표준의 이미지 패턴을 따른다. WebP 패턴의 가운데 네 바이트는
"아무 값"(마스크 00)이다. 이 검사는 앞부분만 보므로 그림 전체가 올바른지는 알 수 없다(README 한계).
"""

import hashlib
import re
import struct
import zlib

# 2 x 1024 x 1024. 크기 한도는 실습에서 정한 값.
MAX_BYTES = 2 * 1024 * 1024  # 2 MiB = 2,097,152바이트(실습에서 정한 값)
MAX_NAME_LEN = 100
# 허용 글자: 영문·숫자·한글·공백·_ . -. 경로 구분자(/ \)가 없다는 것이 핵심.
NAME_OK = re.compile(r"^[A-Za-z0-9가-힣 _.-]+$")

# 확장자 -> 기대 MIME. svg, html은 스크립트를 담을 수 있어 목록에 없다.
ALLOWED = {  # 확장자 -> 기대하는 형식
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}

# (형식, 패턴, 마스크) : 마스크가 00인 자리는 아무 값이나 된다
# (형식, 패턴, 마스크). 마스크의 FF 자리는 패턴과 같아야 하고 00 자리는 무엇이든 된다.
SIGNATURES = [
    ("image/png", bytes.fromhex("89504E470D0A1A0A"), bytes.fromhex("FFFFFFFFFFFFFFFF")),
    ("image/jpeg", bytes.fromhex("FFD8FF"), bytes.fromhex("FFFFFF")),
    # WebP: 'RIFF' + 파일 크기 4바이트(아무 값) + 'WEBPVP8 '. 가운데 4바이트가 마스크 00.
    ("image/webp", bytes.fromhex("52494646000000005745425056 50".replace(" ", "")),
     bytes.fromhex("FFFFFFFF00000000FFFFFFFFFFFF")),
]


def sniff(data):
    """앞부분 바이트로 형식을 판정한다. 모르면 None."""
    for mime, pat, mask in SIGNATURES:
        # (data[i] & mask[i]) == pat[i]: 마스크 자리만 비교. 데이터가 패턴보다 짧으면 비교하지 않는다.
        if len(data) >= len(pat) and all((data[i] & mask[i]) == pat[i] for i in range(len(pat))):
            return mime
    return None


def last_extension(name):
    # 윈도 경로 구분자도 / 로 통일한 뒤 마지막 조각만 본다. 확장자는 마지막 점 뒤만 사용(card.png.html -> html).
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    return base.rsplit(".", 1)[-1].lower() if "." in base else ""


def validate(filename, content_type, data):
    """검사 결과 목록과 최종 판정. 모든 검사를 끝까지 돌려 어디서 걸렸는지 다 적는다."""
    reasons = []
    if len(data) == 0:
        reasons.append("빈 파일")
    if len(data) > MAX_BYTES:
        reasons.append("크기 초과")
    # 이름 길이, 허용 글자, 상위 폴더(..) 포함 여부를 한꺼번에 본다.
    if len(filename) > MAX_NAME_LEN or not NAME_OK.match(filename) or ".." in filename:
        reasons.append("이름 규칙 위반")
    ext = last_extension(filename)
    expected = ALLOWED.get(ext)
    if expected is None:
        reasons.append("허용하지 않는 확장자")
    # Content-Type 뒤의 ; charset=... 부분은 떼고 소문자로 비교.
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct not in set(ALLOWED.values()):
        reasons.append("허용하지 않는 Content-Type")
    elif expected and ct != expected:
        reasons.append("Content-Type과 확장자 불일치")
    # 최종 판단 근거는 내용이다. Content-Type은 브라우저가 적은 말일 뿐이라 참고용.
    detected = sniff(data)
    if expected and detected != expected:
        reasons.append("매직 바이트 불일치")
    # 사유 목록이 비어 있어야 통과. 모든 검사를 끝까지 돌려 사유를 전부 남긴다.
    return {"ok": not reasons, "reasons": reasons, "detected": detected}


def stored_name(data, detected):
    """서버가 만드는 저장 이름: 내용의 해시 앞 16자리 + 판정한 형식의 확장자."""
    # 저장 이름의 확장자는 사용자가 준 값이 아니라 판정한 형식에서 얻는다.
    ext = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}[detected]
    return hashlib.sha256(data).hexdigest()[:16] + "." + ext


# ---------- 시험용 파일(모두 이 장에서 만든 것) ----------

def tiny_png(w=1, h=1, rgb=(47, 93, 138)):
    """표준 라이브러리로 만든 진짜 PNG(1×1)."""
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))
    # PNG는 한 줄마다 앞에 필터 종류 1바이트(0 = 없음)가 붙는다. 그 뒤 픽셀(RGB 3바이트) x 가로.
    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    # IHDR: 가로, 세로, 비트 깊이 8, 색 유형 2(RGB), 압축 0, 필터 0, 인터레이스 0. 청크 끝은 CRC-32(종류+본문).
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def jpeg_head():
    """JPEG 머리(SOI + APP0 JFIF)만 있는 자리표시. 앞부분 검사만 통과하고 그림으로는 열리지 않는다."""
    return bytes.fromhex("FFD8FFE000104A46494600010100000100010000") + b"\x00" * 16 + bytes.fromhex("FFD9")


def webp_head():
    """WebP 머리(RIFF....WEBPVP8 )만 있는 자리표시."""
    body = b"WEBPVP8 " + struct.pack("<I", 10) + b"\x00" * 10
    return b"RIFF" + struct.pack("<I", len(body)) + body


HTML_TEXT = "<!doctype html><title>메모</title><p>그림이 아닌 문서예요</p>".encode()
SVG_TEXT = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"><rect width="10" height="10"/></svg>'


def samples():
    """(이름표, 파일 이름, 브라우저가 보낸 Content-Type, 내용). 순서는 결과 표 순서."""
    png = tiny_png()
    return [
        ("정상 PNG", "autumn-bg.png", "image/png", png),
        ("정상 JPEG 머리", "photo.jpg", "image/jpeg", jpeg_head()),
        ("정상 WebP 머리", "pattern.webp", "image/webp", webp_head()),
        ("2 MiB보다 1바이트 큰 PNG", "huge.png", "image/png", png + b"\x00" * (MAX_BYTES + 1 - len(png))),
        ("HTML 문서", "notes.html", "text/html", HTML_TEXT),
        ("이름만 .png인 HTML 문서", "notes.png", "image/png", HTML_TEXT),
        ("SVG 그림", "logo.svg", "image/svg+xml", SVG_TEXT),
        ("확장자가 두 개(.png.html)", "card.png.html", "text/html", HTML_TEXT),
        ("이름에 폴더 이동(../)이 든 PNG", "../server_only/bg.png", "image/png", png),
        ("Content-Type이 다른 진짜 PNG", "bg2.png", "application/octet-stream", png),
        ("빈 파일", "empty.png", "image/png", b""),
    ]
