"""Capacitor 설정 점검기 (1인 개발 앱 출시 실전 7장).

project 폴더 하나(capacitor.config.json, AndroidManifest.xml, build.gradle, variables.gradle,
app_features.json, gitignore.txt)를 읽고 출시 전에 고칠 곳을 찾는다.

    uv run python webapp/ch07_capacitor_apk/webapp_ch07_config.py <폴더>

판정은 두 가지다.
- "막음": 릴리스 APK로 내보내면 안 되는 것. 하나라도 있으면 종료 코드 1.
- "확인": 사람이 이유를 적어 두면 넘어갈 수 있는 것.
규칙마다 data/sources_2026-10-10.json의 사실 id를 붙인다. 표준 라이브러리만 쓴다.
"""

import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).parent
SOURCES = HERE / "data" / "sources_2026-10-10.json"
ANDROID = "{http://schemas.android.com/apk/res/android}"

# Application ID 규칙: 점으로 나뉜 두 토막 이상, 토막마다 글자로 시작, 글자·숫자·밑줄만
APP_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+$")

# 플러그인 문서가 요구하는 권한(조건부). 여기 없는 위험 권한은 "쓰는 곳 없음"으로 본다.
STORAGE = {"android.permission.READ_EXTERNAL_STORAGE": 32, "android.permission.WRITE_EXTERNAL_STORAGE": 29}
ALWAYS_OK = {"android.permission.INTERNET"}

CAPACITOR_MIN_SDK = 24
PLAY_MAX_VERSION_CODE = 2_100_000_000
RECOMMENDED_KEY_YEARS = 25
SECRET_FILE_SUFFIXES = (".jks", ".keystore")


def load_sources():
    return {f["id"]: f for f in json.loads(SOURCES.read_text(encoding="utf-8"))["facts"]}


def _line_of(text, needle):
    for i, line in enumerate(text.splitlines(), 1):
        if needle in line:
            return i
    return None


def _finding(rule, level, file, line, message, source):
    return {"rule": rule, "level": level, "file": file, "line": line, "message": message, "source": source}


def check_capacitor_config(folder):
    path = folder / "capacitor.config.json"
    text = path.read_text(encoding="utf-8")
    cfg = json.loads(text)
    out = []
    app_id = cfg.get("appId", "")
    if not APP_ID.match(app_id):
        out.append(_finding("app_id_format", "막음", path.name, _line_of(text, '"appId"'),
                            f"appId '{app_id}'가 Application ID 규칙(두 토막 이상, 글자로 시작, 글자·숫자·밑줄)에 맞지 않아요",
                            "android_app_id"))
    server = cfg.get("server", {})
    if "url" in server:
        out.append(_finding("server_url", "막음", path.name, _line_of(text, '"url"'),
                            f"server.url({server['url']})은 개발 중 라이브 리로드용이에요. 릴리스에서는 지워요",
                            "cap_server_url"))
    if server.get("cleartext") is True:
        out.append(_finding("server_cleartext", "막음", path.name, _line_of(text, '"cleartext"'),
                            "server.cleartext: true는 암호화 안 된 HTTP를 허용해요. 운영용이 아니에요",
                            "cap_server_cleartext"))
    scheme = server.get("androidScheme", "https")
    if scheme != "https":
        out.append(_finding("android_scheme", "확인", path.name, _line_of(text, '"androidScheme"'),
                            f"androidScheme이 기본값 https가 아니라 '{scheme}'예요. 바꾼 이유를 적어 둬요",
                            "cap_android_scheme"))
    android = cfg.get("android", {})
    if android.get("allowMixedContent") is True:
        out.append(_finding("mixed_content", "막음", path.name, _line_of(text, '"allowMixedContent"'),
                            "allowMixedContent: true는 https 화면 안에서 http 자원을 섞어 받게 해요",
                            "cap_mixed_content"))
    if android.get("webContentsDebuggingEnabled") is True:
        out.append(_finding("webview_debug", "막음", path.name, _line_of(text, '"webContentsDebuggingEnabled"'),
                            "webContentsDebuggingEnabled: true면 릴리스 앱의 WebView를 개발자 도구로 열 수 있어요",
                            "cap_webview_debug"))
    return cfg, out


def check_manifest(folder, features):
    path = folder / "AndroidManifest.xml"
    text = path.read_text(encoding="utf-8")
    root = ET.fromstring(text)
    out = []
    app = root.find("application")
    if app.get(ANDROID + "usesCleartextTraffic") == "true":
        out.append(_finding("manifest_cleartext", "막음", path.name, _line_of(text, "usesCleartextTraffic"),
                            "usesCleartextTraffic=\"true\"는 Android 9부터 꺼져 있던 HTTP를 다시 켜요",
                            "android_cleartext_default"))
    if app.get(ANDROID + "debuggable") == "true":
        out.append(_finding("manifest_debuggable", "막음", path.name, _line_of(text, "debuggable"),
                            "debuggable을 직접 적지 말아요. 개발 빌드에는 도구가 알아서 붙여요",
                            "android_debuggable"))
    for act in app.findall("activity"):
        if act.find("intent-filter") is not None and act.get(ANDROID + "exported") is None:
            out.append(_finding("activity_exported", "막음", path.name, _line_of(text, act.get(ANDROID + "name")),
                                f"intent-filter가 있는 {act.get(ANDROID + 'name')}에 android:exported가 없어요",
                                "android_exported"))
    perms = []
    for p in root.findall("uses-permission"):
        name = p.get(ANDROID + "name")
        max_sdk = p.get(ANDROID + "maxSdkVersion")
        perms.append({"name": name, "maxSdkVersion": int(max_sdk) if max_sdk else None})
        line = _line_of(text, name + '"')
        if name in ALWAYS_OK:
            continue
        if name in STORAGE:
            needed = features["camera"]["save_to_gallery"] or any(
                d in ("Documents", "ExternalStorage") for d in features["filesystem"]["directories"])
            if not needed:
                out.append(_finding("permission_unused", "확인", path.name, line,
                                    f"{name}: 갤러리 저장도, Documents·ExternalStorage도 안 쓰는데 넣었어요",
                                    "cap_camera_perms"))
            elif max_sdk is None:
                out.append(_finding("permission_no_max_sdk", "확인", path.name, line,
                                    f"{name}에 android:maxSdkVersion=\"{STORAGE[name]}\"를 붙여 새 기기에서는 요청하지 않게 해요",
                                    "cap_camera_perms"))
            continue
        out.append(_finding("permission_unused", "확인", path.name, line,
                            f"{name}: 쓰는 플러그인({', '.join(features['plugins'])})의 문서가 요구하지 않는 권한이에요",
                            "cap_camera_perms"))
    return perms, out


def read_gradle(folder):
    build = (folder / "build.gradle").read_text(encoding="utf-8")
    variables = (folder / "variables.gradle").read_text(encoding="utf-8")

    def grab(pattern, text, cast=str):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    return {
        "applicationId": grab(r'applicationId\s+"([^"]+)"', build),
        "versionCode": grab(r"versionCode\s+(\d+)", build, int),
        "versionName": grab(r'versionName\s+"([^"]+)"', build),
        "minSdkVersion": grab(r"minSdkVersion\s*=\s*(\d+)", variables, int),
        "targetSdkVersion": grab(r"targetSdkVersion\s*=\s*(\d+)", variables, int),
        "password_literals": len(re.findall(r'(?:store|key)Password\s+"[^"]+"', build)),
        "_build": build,
    }


def check_gradle(folder, cfg):
    g = read_gradle(folder)
    out = []
    if g["minSdkVersion"] is not None and g["minSdkVersion"] < CAPACITOR_MIN_SDK:
        out.append(_finding("min_sdk", "막음", "variables.gradle", _line_of((folder / "variables.gradle").read_text(encoding="utf-8"), "minSdkVersion"),
                            f"minSdkVersion {g['minSdkVersion']}: Capacitor 8은 {CAPACITOR_MIN_SDK} 이상이에요",
                            "cap8_sdk"))
    if g["applicationId"] != cfg.get("appId"):
        out.append(_finding("app_id_mismatch", "확인", "build.gradle", _line_of(g["_build"], "applicationId"),
                            f"build.gradle applicationId({g['applicationId']})와 capacitor appId({cfg.get('appId')})가 달라요",
                            "android_app_id"))
    vc = g["versionCode"]
    if vc is None or not (0 < vc <= PLAY_MAX_VERSION_CODE):
        out.append(_finding("version_code_range", "막음", "build.gradle", _line_of(g["_build"], "versionCode"),
                            f"versionCode {vc}는 1 이상 {PLAY_MAX_VERSION_CODE} 이하의 정수여야 해요",
                            "android_versioncode"))
    elif encode_version(g["versionName"]) not in (None, vc):
        out.append(_finding("version_code_rule", "확인", "build.gradle", _line_of(g["_build"], "versionCode"),
                            f"versionName {g['versionName']}이면 예시 규칙으로는 versionCode {encode_version(g['versionName'])}인데 {vc}예요",
                            "android_versioncode"))
    if g["password_literals"]:
        out.append(_finding("password_in_gradle", "막음", "build.gradle", _line_of(g["_build"], "storePassword"),
                            f"build.gradle에 비밀번호 글자 {g['password_literals']}개가 그대로 있어요. keystore.properties로 빼요",
                            "android_no_password_in_gradle"))
    return g, out


def check_secrets_in_folder(folder):
    out = []
    ignore = (folder / "gitignore.txt").read_text(encoding="utf-8").split()
    for p in sorted(folder.iterdir()):
        if p.suffix in SECRET_FILE_SUFFIXES:
            out.append(_finding("keystore_in_repo", "막음", p.name, None,
                                "키 저장소 파일이 프로젝트 폴더 안에 있어요. 저장소 밖으로 옮기고 백업해요",
                                "android_key_lost"))
        if p.name == "keystore.properties" and "keystore.properties" not in ignore:
            out.append(_finding("properties_not_ignored", "막음", p.name, None,
                                "keystore.properties가 .gitignore에 없어요", "android_no_password_in_gradle"))
    return out


def key_years(days):
    return days / 365.25


def check_key_validity(features):
    days = features.get("keytool_validity_days")
    if days is None:
        return []
    if key_years(days) < RECOMMENDED_KEY_YEARS:
        return [_finding("key_validity", "확인", "app_features.json", None,
                         f"키 유효기간 {days}일은 {key_years(days):.1f}년이에요. 25년 이상을 권해요",
                         "android_key_validity")]
    return []


def encode_version(name):
    """예시 규칙: versionCode = 큰 변경 × 10000 + 중간 변경 × 100 + 작은 수정 (중간·작은 자리는 0~99)."""
    if name is None:
        return None
    parts = name.split(".")
    if len(parts) != 3 or not all(x.isdigit() for x in parts):
        return None
    major, minor, patch = map(int, parts)
    if minor > 99 or patch > 99:
        raise ValueError("중간·작은 자리는 0~99만 써요")
    return major * 10000 + minor * 100 + patch


def check_project(folder):
    folder = Path(folder)
    features = json.loads((folder / "app_features.json").read_text(encoding="utf-8"))
    cfg, f1 = check_capacitor_config(folder)
    perms, f2 = check_manifest(folder, features)
    gradle, f3 = check_gradle(folder, cfg)
    f4 = check_secrets_in_folder(folder)
    f5 = check_key_validity(features)
    findings = f1 + f2 + f3 + f4 + f5
    gradle.pop("_build")
    return {
        "folder": folder.name,
        "appId": cfg.get("appId"),
        "permissions": perms,
        "gradle": gradle,
        "findings": findings,
        "blocking": sum(1 for f in findings if f["level"] == "막음"),
        "review": sum(1 for f in findings if f["level"] == "확인"),
    }


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    res = check_project(argv[1])
    for f in res["findings"]:
        where = f"{f['file']}:{f['line']}" if f["line"] else f["file"]
        print(f"{f['level']}  {where:28s} {f['rule']:22s} {f['message']}")
    print(f"막음 {res['blocking']}건, 확인 {res['review']}건")
    return 1 if res["blocking"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
