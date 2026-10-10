"""스토어 없이 APK를 나눠 줄 때의 설치·업데이트 모형 (1인 개발 앱 출시 실전 7장).

진짜 Android가 아니다. 공식 문서에 적힌 규칙 네 가지만 흉내 낸다.
1. 기기의 API 수준이 APK의 minSdk보다 낮으면 설치하지 않는다 (android_versioncode의 minSdk 정의)
2. 같은 패키지가 이미 있으면 서명 인증서가 같을 때만 업데이트한다 (android_signing_required)
3. 이미 있는 것보다 versionCode가 낮으면 설치하지 않는다 (android_versioncode)
4. Play 밖 파일은 그 출처(앱)에 '알 수 없는 앱 설치'를 허용해야 설치된다 (android_unknown_apps, Android 8.0+)

같은 versionCode로 다시 설치하는 경우는 오늘 읽은 문서에서 기기 동작을 확인하지 못해 'unknown'으로 돌려준다.
삭제하면 앱 전용 데이터가 함께 지워진다고 둔다([해석], 이 장에서 출처로 확인하지 않음).
인증서는 진짜 키가 아니라 이름표 문자열의 SHA-256으로 흉내 낸다.
"""

import hashlib
from dataclasses import dataclass, field


def fake_cert(label):
    """가짜 인증서 지문. 진짜 키가 아니라 이름표 문자열의 SHA-256 앞 16글자."""
    return "DEMO-" + hashlib.sha256(label.encode()).hexdigest()[:16]


RELEASE = fake_cert("DEMO event-order release key 2026")
DEBUG_LAPTOP_B = fake_cert("DEMO debug key on teammate laptop")
NEW_AFTER_LOSS = fake_cert("DEMO new key made after losing the old one")


@dataclass(frozen=True)
class Apk:
    package: str
    version_name: str
    version_code: int
    cert: str
    min_sdk: int = 24


@dataclass
class Device:
    name: str
    api_level: int
    unknown_sources_allowed: set = field(default_factory=set)
    installed: dict = field(default_factory=dict)  # package -> (Apk, 데이터 dict)

    def install(self, apk, source):
        if source != "adb" and source not in self.unknown_sources_allowed:
            return "refused_unknown_source"
        if self.api_level < apk.min_sdk:
            return "refused_min_sdk"
        cur = self.installed.get(apk.package)
        if cur is None:
            self.installed[apk.package] = (apk, {})
            return "installed"
        old, data = cur
        if old.cert != apk.cert:
            return "refused_signature"
        if apk.version_code < old.version_code:
            return "refused_downgrade"
        if apk.version_code == old.version_code:
            return "unknown_same_code"
        self.installed[apk.package] = (apk, data)  # 업데이트는 데이터를 남긴다
        return "updated"

    def uninstall(self, package):
        had = self.installed.pop(package, None)
        return 0 if had is None else len(had[1])  # 지워진 데이터 항목 수


def fleet(n=8, api_level=33):
    """행사장 테이블 태블릿 n대(가상). 첫 설치 전에 파일 관리자 앱에만 출처 허용을 켠다."""
    return [Device(f"T{i + 1:02d}", api_level, {"files"}) for i in range(n)]


def run_scenario():
    """가상의 행사장 주문 앱 하루. 단계마다 기기별 결과를 센다."""
    pkg = "invalid.demo.eventorder"
    tabs = fleet()
    spare = Device("SPARE", 23, {"files"})  # 창고의 옛 태블릿(Android 6.0, API 23)
    steps = []

    def push(label, apk, devices, source="files"):
        res = [d.install(apk, source) for d in devices]
        counts = {}
        for r in res:
            counts[r] = counts.get(r, 0) + 1
        steps.append({"step": label, "version_name": apk.version_name, "version_code": apk.version_code,
                      "cert": apk.cert, "devices": len(devices), "result": counts})
        return res

    v100 = Apk(pkg, "1.0.0", 10000, RELEASE)
    push("S1 행사 전날 첫 설치", v100, tabs)
    for d in tabs:  # 주문 앱이 기기에 남기는 작은 설정(테이블 번호 등)
        d.installed[pkg][1]["table_no"] = d.name
    push("S1b 창고 태블릿에도 설치", v100, [spare])
    push("S2 오타 고친 1.0.1", Apk(pkg, "1.0.1", 10001, RELEASE), tabs)
    push("S3 팀원 노트북 디버그 빌드 1.0.2", Apk(pkg, "1.0.2", 10002, DEBUG_LAPTOP_B), tabs)
    push("S4 되돌리려고 1.0.0 다시 빌드", Apk(pkg, "1.0.0", 10000, RELEASE), tabs)
    push("S4b 옛 코드를 1.0.3으로 올려 배포", Apk(pkg, "1.0.3", 10003, RELEASE), tabs)
    push("S4c 번호 안 올리고 다시 배포", Apk(pkg, "1.0.3", 10003, RELEASE), tabs)
    push("S5 키를 잃고 새 키로 1.1.0", Apk(pkg, "1.1.0", 10100, NEW_AFTER_LOSS), tabs)
    lost = sum(d.uninstall(pkg) for d in tabs)
    push("S5b 모두 지우고 새 키로 다시 설치", Apk(pkg, "1.1.0", 10100, NEW_AFTER_LOSS), tabs)
    kept = sum(1 for d in tabs if d.installed[pkg][1].get("table_no"))
    return {"package": pkg, "tablets": len(tabs), "steps": steps,
            "data_items_lost_on_reinstall": lost, "tablets_with_table_no_after_reinstall": kept,
            "manual_touches_for_key_loss": 2 * len(tabs)}


# 무엇을 바꾸면 새 APK가 필요한가 (이 장의 정리, 근거 id와 함께)
CHANGE_TABLE = [
    {"change": "메뉴 이름·가격(서버 DB의 행)", "new_apk": False, "why": "앱이 실행 중에 서버에서 읽어 오는 데이터", "source": "[해석]"},
    {"change": "화면 글자·CSS·JS(webDir에 묶은 웹 파일)", "new_apk": True, "why": "APK 안에 들어 있는 파일. 외부 라이브 업데이트 서비스를 쓰면 예외가 생길 수 있음", "source": "cap_config_core, play_self_update_policy"},
    {"change": "capacitor.config.json", "new_apk": True, "why": "빌드할 때 네이티브 프로젝트로 들어감", "source": "[해석]"},
    {"change": "새 플러그인·권한", "new_apk": True, "why": "네이티브 코드와 AndroidManifest가 바뀜", "source": "cap_camera_perms"},
    {"change": "서명 키", "new_apk": True, "why": "인증서가 다르면 업데이트가 안 되어 삭제 후 재설치", "source": "android_signing_required"},
]


def version_code_examples():
    from webapp_ch07_config import encode_version
    names = ["1.0.0", "1.0.1", "1.0.3", "1.1.0", "1.2.3", "2.0.0"]
    return {n: encode_version(n) for n in names}


def plan_is_increasing(codes):
    """배포 계획의 versionCode가 엄격히 늘어나는지. 처음 어긋난 자리(0부터)나 None."""
    for i in range(1, len(codes)):
        if codes[i] <= codes[i - 1]:
            return i
    return None
