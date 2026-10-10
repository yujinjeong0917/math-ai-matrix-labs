"""서명 키 관리 점검표 (1인 개발 앱 출시 실전 7장).

항목 10개 가운데 6개는 폴더만 보고 기계가 판정하고, 4개는 사람이 답한다.
사람 항목은 checks/ch07.csv에 직접 적는다. 이 모듈은 두 예시 폴더의 기계 항목만 채운다.
"""

import hashlib
import re
from pathlib import Path

import webapp_ch07_config as C

ITEMS = [
    # id, 누가 판정, 질문, 근거 id
    ("store_outside_repo", "기계", "키 저장소 파일(.jks/.keystore)이 프로젝트 폴더 밖에 있나", "android_key_lost"),
    ("properties_ignored", "기계", "keystore.properties가 .gitignore에 들어 있나(파일이 있다면)", "android_no_password_in_gradle"),
    ("no_password_in_gradle", "기계", "build.gradle에 비밀번호 글자가 없나", "android_no_password_in_gradle"),
    ("validity_25y", "기계", "키 유효기간이 25년 이상인가", "android_key_validity"),
    ("release_not_debuggable", "기계", "릴리스 설정에 debuggable·WebView 디버깅이 꺼져 있나", "cap_webview_debug"),
    ("version_code_ok", "기계", "versionCode가 1 이상 2100000000 이하이고 규칙대로 붙었나", "android_versioncode"),
    ("backup_two_places", "사람", "키 파일과 비밀번호를 서로 다른 두 곳 이상(하나는 오프라인)에 따로 백업했나", "android_key_lost"),
    ("fingerprint_recorded", "사람", "인증서 SHA-256 지문을 적어 두고, 배포 전에 apksigner verify로 대조하나", "android_build_cmdline"),
    ("owner_named", "사람", "키를 가진 사람과 잃었을 때 할 일을 적어 두었나", "android_key_lost"),
    ("no_debug_key_release", "사람", "나눠 주는 APK가 디버그 키가 아닌 릴리스 키로 서명됐나", "android_build_cmdline"),
]


def machine_answers(folder):
    res = C.check_project(folder)
    rules = {f["rule"] for f in res["findings"]}
    return {
        "store_outside_repo": "keystore_in_repo" not in rules,
        "properties_ignored": "properties_not_ignored" not in rules,
        "no_password_in_gradle": "password_in_gradle" not in rules,
        "validity_25y": "key_validity" not in rules,
        "release_not_debuggable": not ({"webview_debug", "manifest_debuggable"} & rules),
        "version_code_ok": not ({"version_code_range", "version_code_rule"} & rules),
    }


def scorecard(folder):
    ans = machine_answers(folder)
    rows = []
    for item_id, who, question, src in ITEMS:
        rows.append({"id": item_id, "who": who, "question": question, "source": src,
                     "pass": ans.get(item_id) if who == "기계" else None})
    machine = [r for r in rows if r["who"] == "기계"]
    return {"rows": rows, "machine_pass": sum(1 for r in machine if r["pass"]),
            "machine_total": len(machine), "human_total": len(rows) - len(machine)}


def keytool_years(days):
    return round(C.key_years(days), 1)


def sha256_file_bytes(data):
    return hashlib.sha256(data).hexdigest()


def demo_apk_bytes(version):
    """진짜 APK가 아니라 '나눠 줄 파일'을 흉내 낸 바이트. 배포 링크 옆에 SHA-256을 함께 적는 연습용."""
    return f"DEMO NOT A REAL APK invalid.demo.eventorder {version}\n".encode()


def no_real_secrets(folder):
    """가짜 값 규칙: 비밀번호 칸은 DEMO_로 시작하거나 '(직접 적기)'만 허용."""
    bad = []
    for p in Path(folder).iterdir():
        if p.is_file():
            for m in re.finditer(r'(?:storePassword|keyPassword)\s*[=\s]\s*"?([^"\n]+)"?', p.read_text(encoding="utf-8")):
                v = m.group(1).strip()
                if not (v.startswith("DEMO_") or v == "(직접 적기)" or v.startswith("keystoreProperties")):
                    bad.append((p.name, v))
    return bad
