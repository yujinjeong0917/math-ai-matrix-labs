"""1인 개발 앱 출시 실전 7장 실험. `uv run python webapp/ch07_capacitor_apk/experiments.py` 로 실행한다.

E0 설정 파일 크기와 줄 수(바른 판·잘못 고친 판)
E1 설정 점검기: 두 판의 막음·확인 건수와 규칙 목록
E2 권한: 두 판이 적은 권한과 점검 결과
E3 서명 점검표: 기계 항목 6개 결과, keytool 유효기간 손계산
E4 versionCode 예시 규칙과 배포 계획 검사
E5 행사장 주문 앱(가상) 하루: 태블릿 8대 설치·업데이트 모형
E6 배포 파일 SHA-256: 원본과 한 글자 바뀐 파일
E7 무엇을 바꾸면 새 APK가 필요한가

표준 라이브러리만 쓴다. 네트워크·Android SDK 없이 돌고 결과는 매번 같다. 실제 APK는 만들지 않는다.
"""

import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import webapp_ch07_config as C  # noqa: E402
import webapp_ch07_signing as G  # noqa: E402
import webapp_ch07_update as U  # noqa: E402

OUT = HERE / "results" / "ch07.json"
GOOD = HERE / "project" / "cardnews_good"
MISTAKE = HERE / "project" / "cardnews_mistake"


def e0_files():
    out = {}
    for folder in (GOOD, MISTAKE):
        out[folder.name] = {p.name: {"bytes": p.stat().st_size, "lines": len(p.read_text(encoding="utf-8").splitlines())}
                            for p in sorted(folder.iterdir()) if p.is_file()}
    return out


def e1_checker():
    out = {}
    for folder in (GOOD, MISTAKE):
        r = C.check_project(folder)
        out[folder.name] = {"blocking": r["blocking"], "review": r["review"],
                            "rules": [f["rule"] for f in r["findings"]],
                            "findings": r["findings"], "gradle": r["gradle"], "appId": r["appId"]}
    return out


def e2_permissions(e1):
    out = {}
    for folder in (GOOD, MISTAKE):
        r = C.check_project(folder)
        flagged = [f for f in r["findings"] if f["rule"].startswith("permission")]
        out[folder.name] = {"declared": [p["name"].split(".")[-1] for p in r["permissions"]],
                            "declared_count": len(r["permissions"]), "flagged": len(flagged),
                            "kept": len(r["permissions"]) - len(flagged)}
    return out


def e3_signing():
    out = {}
    for folder in (GOOD, MISTAKE):
        s = G.scorecard(folder)
        out[folder.name] = {"machine_pass": s["machine_pass"], "machine_total": s["machine_total"],
                            "human_total": s["human_total"],
                            "failed": [r["id"] for r in s["rows"] if r["pass"] is False]}
    out["keytool_years"] = {"10000": G.keytool_years(10000), "3650": G.keytool_years(3650),
                            "days_for_25y": round(25 * 365.25)}
    return out


def e4_versions():
    ex = U.version_code_examples()
    plan_ok = [ex[n] for n in ("1.0.0", "1.0.1", "1.0.3", "1.1.0")]
    plan_bad = [10000, 10001, 10000, 10003]
    return {"rule": "큰 변경 × 10000 + 중간 변경 × 100 + 작은 수정", "examples": ex,
            "plan_ok": plan_ok, "plan_ok_first_break": U.plan_is_increasing(plan_ok),
            "plan_bad": plan_bad, "plan_bad_first_break": U.plan_is_increasing(plan_bad),
            "play_max": C.PLAY_MAX_VERSION_CODE, "max_major_under_play": C.PLAY_MAX_VERSION_CODE // 10000}


def e6_hash():
    a = G.demo_apk_bytes("1.0.1")
    b = bytearray(a)
    b[-2] = ord("2")  # 마지막 숫자 한 글자만 바꾼 파일
    ha, hb = G.sha256_file_bytes(a), G.sha256_file_bytes(bytes(b))
    same_pos = sum(1 for x, y in zip(ha, hb) if x == y)
    return {"bytes": len(a), "sha256_original": ha, "sha256_one_char_changed": hb,
            "hex_digits_same_position": same_pos, "hex_digits": len(ha)}


def main():
    e1 = e1_checker()
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "checked": "2026-10-10", "note": "실제 APK를 만들지 않는 모형 값이에요. 키·비밀번호·주소는 모두 가짜예요."},
        "e0_files": e0_files(),
        "e1_checker": e1,
        "e2_permissions": e2_permissions(e1),
        "e3_signing": e3_signing(),
        "e4_versions": e4_versions(),
        "e5_event_fleet": U.run_scenario(),
        "e6_hash": e6_hash(),
        "e7_change_table": U.CHANGE_TABLE,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
