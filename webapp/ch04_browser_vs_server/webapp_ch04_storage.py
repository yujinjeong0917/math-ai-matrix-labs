"""브라우저 저장소에 무엇이 얼마나 들어가고, 언제 지워지는지 세는 모형.

출처: MDN "Storage quotas and eviction criteria"(2026-10-10 확인).
- localStorage는 출처(origin)마다 5 MiB. 값은 UTF-16 문자열이라, 이 모형은 글자 하나를 2바이트로 센다(가정).
- 그림을 localStorage에 넣으려면 문자열로 바꿔야 해서 base64(3바이트 → 4글자)를 쓴다고 둔다.
- IndexedDB는 파일·Blob을 그대로 넣을 수 있고, 한도는 디스크 크기의 비율로 정해진다.
- 지워지는 조건: 저장 공간이 모자라면 가장 오래 안 쓴 출처부터(LRU), Safari는 추적 방지가 켜져 있을 때
  Safari를 쓴 날로 최근 7일 동안 그 사이트에 사용자 상호작용이 없으면 스크립트가 만든 데이터를, 사생활 보호 모드는 끝날 때.
  영구 저장(persist)을 받으면 사용자가 직접 지울 때만 지워진다.
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).parent
DRAFT = HERE / "data" / "draft_example.json"
MIB = 1024 * 1024
LOCAL_STORAGE_BYTES = 5 * MIB

QUOTA_SHARE = {  # 출처(origin) 하나가 쓸 수 있는 디스크 비율, MDN 기준
    "Chrome·Edge(Chromium)": 0.60,
    "Safari 브라우저 앱(macOS 14, iOS 17부터)": 0.60,
    "웹 화면을 품은 다른 WebKit 앱": 0.15,
}
FIREFOX_BEST_EFFORT = {"share": 0.10, "group_cap_gib": 10}


def utf16_units(s):
    """UTF-16으로 셌을 때의 글자(코드 단위) 수. 한글·영문은 1, 이모지 같은 보조 평면 문자는 2."""
    return len(s.encode("utf-16-le")) // 2


def base64_chars(nbytes):
    return 4 * math.ceil(nbytes / 3)


def local_storage_fit(item_chars, key_chars, bytes_per_char=2, limit=LOCAL_STORAGE_BYTES):
    per_item = (item_chars + key_chars) * bytes_per_char
    return limit // per_item, per_item


def draft_text():
    return DRAFT.read_text(encoding="utf-8").strip()


def evicted(*, browser, persisted=False, storage_pressure_lru_first=False, days_without_interaction=0,
            tracking_prevention=True, private_mode_ended=False):
    """시나리오 하나에서 스크립트가 만든 데이터가 지워지는지와 이유. MDN 문장을 그대로 옮긴 규칙만 쓴다."""
    if private_mode_ended:
        return True, "사생활 보호 모드가 끝남(보통 지워짐)"
    if persisted:
        return False, "영구 저장: 사용자가 설정에서 지울 때만 지워짐"
    if storage_pressure_lru_first:
        return True, "저장 공간 부족, 가장 오래 안 쓴 출처부터 지움(LRU)"
    if browser == "Safari" and tracking_prevention and days_without_interaction >= 7:
        return True, "Safari 추적 방지: Safari를 쓴 최근 7일 동안 클릭·탭이 없음"
    return False, "남아 있음(best-effort)"


SCENARIOS = [
    ("Chrome, 매일 쓰는 사용자", dict(browser="Chrome")),
    ("Chrome, 폰 저장 공간이 꽉 차고 우리 앱을 가장 오래 안 씀", dict(browser="Chrome", storage_pressure_lru_first=True)),
    ("Safari, 매일 쓰지만 우리 앱만 열흘 안 연 사용자", dict(browser="Safari", days_without_interaction=10)),
    ("Safari, 우리 앱만 열흘 안 열었지만 영구 저장을 받음", dict(browser="Safari", days_without_interaction=10, persisted=True)),
    ("아무 브라우저, 사생활 보호 창을 닫음", dict(browser="Chrome", private_mode_ended=True)),
]


def eviction_table():
    return [{"scenario": name, "evicted": evicted(**kw)[0], "why": evicted(**kw)[1]} for name, kw in SCENARIOS]


def idb_cap_bytes(disk_bytes, browser):
    if browser == "Firefox":
        return min(int(disk_bytes * FIREFOX_BEST_EFFORT["share"]), FIREFOX_BEST_EFFORT["group_cap_gib"] * 1024 ** 3)
    return int(disk_bytes * QUOTA_SHARE[browser])
