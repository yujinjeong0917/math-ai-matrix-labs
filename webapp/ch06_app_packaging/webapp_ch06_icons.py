"""1인 개발 앱 출시 실전 6장: 앱 아이콘(PNG)을 표준 라이브러리로 만들고 읽는다.

PNG 파일은 8바이트 서명 + 덩어리(chunk) 여러 개로 되어 있고, 첫 덩어리 IHDR에 가로·세로 픽셀이 들어 있다.
manifest 검사기는 이 값으로 "sizes에 적은 크기와 실제 그림 크기가 같은가"를 확인한다.

    uv run python webapp/ch06_app_packaging/webapp_ch06_icons.py   # icons/ 다시 만들기
"""

import struct
import zlib
from pathlib import Path

HERE = Path(__file__).parent
ICONS = HERE / "pwa" / "public" / "icons"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

BLUE = (0x2F, 0x5D, 0x8A)
WHITE = (0xFF, 0xFF, 0xFF)


def _chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def make_icon(size):
    """파란 바탕 가운데에 흰 카드 한 장. 카드는 가운데 60% 안쪽이라 둥글게 잘려도 남는다."""
    lo, hi = int(size * 0.2), int(size * 0.8)
    rows = []
    for y in range(size):
        row = bytearray([0])  # 줄마다 필터 0(그대로)
        for x in range(size):
            row += bytes(WHITE if lo <= x < hi and lo <= y < hi else BLUE)
        rows.append(bytes(row))
    raw = b"".join(rows)
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8비트 RGB
    return PNG_SIGNATURE + _chunk(b"IHDR", ihdr) + _chunk(b"IDAT", zlib.compress(raw, 9)) + _chunk(b"IEND", b"")


def png_size(data):
    """PNG 바이트에서 (가로, 세로)를 읽는다. PNG가 아니면 None."""
    if len(data) < 24 or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def write_icons():
    ICONS.mkdir(parents=True, exist_ok=True)
    out = {}
    for size in (192, 512):
        data = make_icon(size)
        (ICONS / f"icon-{size}.png").write_bytes(data)
        out[size] = len(data)
    return out


if __name__ == "__main__":
    print(write_icons())
