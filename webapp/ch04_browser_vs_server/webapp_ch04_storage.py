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
# 5 MiB = 5 x 1024 x 1024 = 5,242,880바이트. 브라우저마다 다를 수 있는 값이라 MDN 기준(확인일 2026-10-10)을 따른 가정.
LOCAL_STORAGE_BYTES = 5 * MIB

# IndexedDB 등이 쓸 수 있는 한도의 디스크 대비 비율(MDN 기준). 실제 한도는 브라우저 버전에 따라 바뀔 수 있다.
QUOTA_SHARE = {  # 출처(origin) 하나가 쓸 수 있는 디스크 비율, MDN 기준
    "Chrome·Edge(Chromium)": 0.60,
    "Safari 브라우저 앱(macOS 14, iOS 17부터)": 0.60,
    "웹 화면을 품은 다른 WebKit 앱": 0.15,
}
# Firefox는 디스크의 10%와 그룹 한도 10 GiB 중 작은 쪽(idb_cap_bytes에서 min).
FIREFOX_BEST_EFFORT = {"share": 0.10, "group_cap_gib": 10}


def utf16_units(s):
    """UTF-16으로 셌을 때의 글자(코드 단위) 수. 한글·영문은 1, 이모지 같은 보조 평면 문자는 2."""
    # utf-16-le는 BOM 없이 2바이트 단위로 인코딩한다. 보조 평면 문자(이모지)는 코드 단위 2개.
    return len(s.encode("utf-16-le")) // 2


def base64_chars(nbytes):
    # base64는 3바이트를 4글자로 바꾸고, 모자란 끝은 = 로 채워 4글자 단위로 올림.
    return 4 * math.ceil(nbytes / 3)


def local_storage_fit(item_chars, key_chars, bytes_per_char=2, limit=LOCAL_STORAGE_BYTES):
    # 키 이름도 저장소를 차지하므로 값 길이에 더한다(모형의 단순화: 내부 오버헤드는 무시).
    per_item = (item_chars + key_chars) * bytes_per_char
    # (들어가는 개수, 한 항목의 바이트). 정수 나눗셈이라 나머지 공간은 버린다.
    return limit // per_item, per_item


def draft_text():
    return DRAFT.read_text(encoding="utf-8").strip()


def evicted(*, browser, persisted=False, storage_pressure_lru_first=False, days_without_interaction=0,
            tracking_prevention=True, private_mode_ended=False):
    """시나리오 하나에서 스크립트가 만든 데이터가 지워지는지와 이유. MDN 문장을 그대로 옮긴 규칙만 쓴다."""
    # 조건 확인 순서가 결과를 정한다: 사생활 보호 종료 > 영구 저장 > 저장 공간 부족(LRU) > Safari 7일 규칙.
    if private_mode_ended:
        return True, "사생활 보호 모드가 끝남(보통 지워짐)"
    if persisted:
        return False, "영구 저장: 사용자가 설정에서 지울 때만 지워짐"
    if storage_pressure_lru_first:
        return True, "저장 공간 부족, 가장 오래 안 쓴 출처부터 지움(LRU)"
    # 7일 규칙은 "Safari를 쓴 날 기준 최근 7일"이라 달력 7일과 다를 수 있다(위 docstring). 이 모형은 days_without_interaction을 그 기준의 값으로 받는다.
    if browser == "Safari" and tracking_prevention and days_without_interaction >= 7:
        return True, "Safari 추적 방지: Safari를 쓴 최근 7일 동안 클릭·탭이 없음"
    return False, "남아 있음(best-effort)"


# 시나리오 다섯 개. 기대 결과는 test에서 표로 고정한다.
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
        # Firefox: min(디스크 x 10%, 10 GiB). 128 GB 디스크면 12.8 GB와 10 GiB(약 10.7 GB) 중 후자.
        return min(int(disk_bytes * FIREFOX_BEST_EFFORT["share"]), FIREFOX_BEST_EFFORT["group_cap_gib"] * 1024 ** 3)
    return int(disk_bytes * QUOTA_SHARE[browser])
