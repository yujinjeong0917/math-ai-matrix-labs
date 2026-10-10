"""1인 개발 앱 출시 실전 7장 검증. `uv run pytest -q webapp/ch07_capacitor_apk` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 네트워크·Android SDK 없이 돈다.
"""

import json
from pathlib import Path

import pytest

import webapp_ch07_config as C
import webapp_ch07_signing as G
import webapp_ch07_update as U

HERE = Path(__file__).parent
GOOD = HERE / "project" / "cardnews_good"
MISTAKE = HERE / "project" / "cardnews_mistake"
SOURCES = json.loads((HERE / "data" / "sources_2026-10-10.json").read_text(encoding="utf-8"))
SOURCE_IDS = {f["id"] for f in SOURCES["facts"]}


# ---------- 설정 점검기 ----------

def test_good_project_passes():
    r = C.check_project(GOOD)
    assert r["blocking"] == 0 and r["review"] == 0
    assert r["gradle"] == {"applicationId": "invalid.demo.cardnews", "versionCode": 10001, "versionName": "1.0.1",
                           "minSdkVersion": 24, "targetSdkVersion": 36, "password_literals": 0}


def test_mistake_project_findings():
    r = C.check_project(MISTAKE)
    assert r["blocking"] == 12 and r["review"] == 7
    rules = [f["rule"] for f in r["findings"]]
    for rule in ["app_id_format", "server_url", "server_cleartext", "mixed_content", "webview_debug",
                 "manifest_cleartext", "manifest_debuggable", "activity_exported", "min_sdk",
                 "password_in_gradle", "properties_not_ignored", "keystore_in_repo"]:
        assert rule in rules
    assert rules.count("permission_unused") == 5


def test_every_rule_has_a_source():
    for folder in (GOOD, MISTAKE):
        for f in C.check_project(folder)["findings"]:
            assert f["source"] in SOURCE_IDS, f


@pytest.mark.parametrize("app_id,ok", [
    ("invalid.demo.cardnews", True), ("com.example.myapp", True), ("a.b", True), ("app_1.x_2", True),
    ("cardnews-app", False), ("cardnews", False), ("1app.demo", False), ("com..demo", False),
    ("com.demo.", False), ("com.demo.카드", False), ("com.2demo", False),
])
def test_app_id_rule(app_id, ok):
    assert bool(C.APP_ID.match(app_id)) is ok


def test_checker_exit_codes(capsys):
    assert C.main(["x", str(GOOD)]) == 0
    assert C.main(["x", str(MISTAKE)]) == 1
    assert "막음 12건, 확인 7건" in capsys.readouterr().out


def test_storage_permission_rules(tmp_path):
    # 갤러리에 저장하면 저장소 권한이 필요하고, maxSdkVersion이 없으면 '확인'
    for p in GOOD.iterdir():
        (tmp_path / p.name).write_bytes(p.read_bytes())
    feats = json.loads((tmp_path / "app_features.json").read_text(encoding="utf-8"))
    feats["camera"]["save_to_gallery"] = True
    (tmp_path / "app_features.json").write_text(json.dumps(feats), encoding="utf-8")
    man = (tmp_path / "AndroidManifest.xml").read_text(encoding="utf-8")
    man = man.replace('<uses-permission android:name="android.permission.INTERNET" />',
                      '<uses-permission android:name="android.permission.INTERNET" />\n'
                      '    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE" android:maxSdkVersion="32" />\n'
                      '    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE" />')
    (tmp_path / "AndroidManifest.xml").write_text(man, encoding="utf-8")
    r = C.check_project(tmp_path)
    assert [f["rule"] for f in r["findings"]] == ["permission_no_max_sdk"]
    assert "maxSdkVersion=\"29\"" in r["findings"][0]["message"]


# ---------- 서명 점검표 ----------

def test_signing_scorecard():
    g, m = G.scorecard(GOOD), G.scorecard(MISTAKE)
    assert (g["machine_pass"], g["machine_total"], g["human_total"]) == (6, 6, 4)
    assert m["machine_pass"] == 0


def test_keytool_years_hand_calc():
    assert G.keytool_years(10000) == 27.4      # 10000 ÷ 365.25
    assert G.keytool_years(3650) == 10.0
    assert round(25 * 365.25) == 9131


def test_no_real_secrets():
    for folder in (GOOD, MISTAKE):
        assert G.no_real_secrets(folder) == []
    assert "NOT A REAL KEYSTORE" in (MISTAKE / "release.jks").read_text(encoding="utf-8")


def test_hash_changes_with_one_char():
    a = G.demo_apk_bytes("1.0.1")
    assert G.sha256_file_bytes(a) != G.sha256_file_bytes(G.demo_apk_bytes("1.0.2"))
    assert len(G.sha256_file_bytes(a)) == 64


# ---------- versionCode ----------

def test_version_code_rule():
    assert C.encode_version("1.2.3") == 10203
    assert C.encode_version("2.0.0") == 20000
    assert C.encode_version("1.0") is None
    with pytest.raises(ValueError):
        C.encode_version("1.100.0")
    assert U.plan_is_increasing([10000, 10001, 10003, 10100]) is None
    assert U.plan_is_increasing([10000, 10001, 10000]) == 2


# ---------- 설치·업데이트 모형 ----------

def test_device_rules():
    d = U.Device("T", 33, {"files"})
    a = U.Apk("p.q", "1.0.0", 10, U.RELEASE)
    assert d.install(a, "browser") == "refused_unknown_source"
    assert d.install(a, "files") == "installed"
    assert d.install(U.Apk("p.q", "1.0.1", 11, U.DEBUG_LAPTOP_B), "files") == "refused_signature"
    assert d.install(U.Apk("p.q", "0.9.0", 9, U.RELEASE), "files") == "refused_downgrade"
    assert d.install(U.Apk("p.q", "1.0.1", 11, U.RELEASE), "files") == "updated"
    assert d.install(U.Apk("p.q", "1.0.1", 11, U.RELEASE), "adb") == "unknown_same_code"
    old = U.Device("O", 23, {"files"})
    assert old.install(a, "files") == "refused_min_sdk"


def test_event_scenario():
    s = U.run_scenario()
    res = {st["step"].split()[0]: st["result"] for st in s["steps"]}
    assert res == {"S1": {"installed": 8}, "S1b": {"refused_min_sdk": 1}, "S2": {"updated": 8},
                   "S3": {"refused_signature": 8}, "S4": {"refused_downgrade": 8}, "S4b": {"updated": 8},
                   "S4c": {"unknown_same_code": 8}, "S5": {"refused_signature": 8}, "S5b": {"installed": 8}}
    assert s["data_items_lost_on_reinstall"] == 8
    assert s["tablets_with_table_no_after_reinstall"] == 0
    assert s["manual_touches_for_key_loss"] == 16


def test_fake_certs_differ():
    assert len({U.RELEASE, U.DEBUG_LAPTOP_B, U.NEW_AFTER_LOSS}) == 3
    assert all(c.startswith("DEMO-") for c in (U.RELEASE, U.DEBUG_LAPTOP_B, U.NEW_AFTER_LOSS))


# ---------- 결과 파일과 출처 ----------

def test_results_file_matches():
    res = json.loads((HERE / "results" / "ch07.json").read_text(encoding="utf-8"))
    assert res["e1_checker"]["cardnews_mistake"]["blocking"] == C.check_project(MISTAKE)["blocking"]
    assert res["e2_permissions"]["cardnews_mistake"]["declared_count"] == 6
    assert res["e3_signing"]["keytool_years"]["10000"] == 27.4
    assert res["e4_versions"]["max_major_under_play"] == 210000
    assert res["e5_event_fleet"] == U.run_scenario()
    assert res["e6_hash"]["hex_digits"] == 64


def test_sources_file_has_url_and_date():
    assert SOURCES["checked"] == "2026-10-10"
    for f in SOURCES["facts"]:
        assert f["url"].startswith("https://") and f["checked"] == "2026-10-10"
    assert len(SOURCES["facts"]) == 37
