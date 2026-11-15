"""1인 개발 앱 출시 실전 6장: 네 가지 길 비교표와 조건별 고르기.

순위를 매기지 않는다. 조건(꼭 필요한 것)을 주면 그 조건을 모두 만족하는 길을 표의 순서 그대로 돌려주고,
빠진 길은 어떤 조건에서 빠졌는지 적는다. 값은 data/paths_2026-10-10.json(공식 문서로 확인, 2026-10-10)에서 읽는다.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
PATHS_FILE = HERE / "data" / "paths_2026-10-10.json"


def load_paths():
    return json.loads(PATHS_FILE.read_text(encoding="utf-8"))


def choose(needs, data=None):
    """needs: 조건 id 목록. 돌려주는 값: {"fits": [이름...], "out": {이름: [못 맞춘 조건...]}, "notes": {...}}"""
    data = data or load_paths()
    # 오타 난 조건은 조용히 무시하지 않고 KeyError로 알린다(test가 확인).
    unknown = [n for n in needs if n not in data["conditions"]]
    if unknown:
        raise KeyError(f"모르는 조건: {unknown}")
    fits, out, notes = [], {}, {}
    for p in data["paths"]:
        # 모든 조건을 만족해야 "맞는 길"(AND). 하나라도 못 하면 어떤 조건에서 빠졌는지 목록에 남긴다.
        missing = [n for n in needs if not p["can"][n]]
        if missing:
            out[p["name"]] = missing
        else:
            fits.append(p["name"])
            got = {n: p["notes"][n] for n in needs if n in p["notes"]}
            if got:
                notes[p["name"]] = got
    return {"needs": list(needs), "fits": fits, "out": out, "notes": notes}


# 카드뉴스 편집기에서 나올 법한 조건 묶음(실습에서 정한 예)
CASES = {
    # 조건 묶음의 이름과 조건 id. 기대 결과는 test_cases에서 표로 고정한다.
    "A 링크로 퍼뜨리고, 새 템플릿 알림은 iOS에도": ["reuse_web_code", "ios_push"],
    "B 앱스토어 검색에 나오되 웹 코드는 그대로": ["reuse_web_code", "ios_app_store"],
    "C 행사장 블루투스 프린터에 카드 출력(iOS 포함)": ["ios_bluetooth"],
    "D 심사 없이 오늘 바로 휴대폰에 설치": ["link_install_no_review"],
    "E 웹 코드 그대로 + iOS 블루투스 + 심사 없이": ["reuse_web_code", "ios_bluetooth", "link_install_no_review"],
}


if __name__ == "__main__":
    for name, needs in CASES.items():
        print(name, choose(needs))
