"""1인 개발 앱 출시 실전 6장 검증. `uv run pytest -q webapp/ch06_app_packaging` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
서버는 127.0.0.1의 빈 포트에 잠깐 띄운다. 바깥 네트워크와 계정 없이 돈다.
"""

import json
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

import webapp_ch06_cache as C
import webapp_ch06_icons as I
import webapp_ch06_manifest as M
import webapp_ch06_paths as P
import webapp_ch06_server as S

HERE = Path(__file__).parent
PUBLIC = HERE / "pwa" / "public"
RESULTS = json.loads((HERE / "results" / "ch06.json").read_text(encoding="utf-8"))


def read(rel):
    return (PUBLIC / rel).read_text(encoding="utf-8")


# ---------- manifest ----------

def test_our_manifest_passes_every_check():
    f = M.check_manifest(read("manifest.json"), "http://127.0.0.1:8000/", PUBLIC)
    assert M.failed(f) == []
    assert M.installable_in_chromium(f)
    assert [x["group"] for x in f].count("chromium") == 6


def test_index_links_manifest_and_registers_sw():
    html = read("index.html")
    assert '<link rel="manifest" href="manifest.json" />' in html
    assert 'navigator.serviceWorker.register("/sw.js")' in html
    assert "<!--" not in html  # 1장에서 본 HTML 메모는 지웠다


@pytest.mark.parametrize("name, failed", [
    ("broken_first_draft", ["icons", "start_url", "display"]),
    ("broken_prefer_store", ["prefer_related_applications"]),
    ("broken_trailing_comma", ["json"]),
    ("broken_wrong_size", ["icon1:icons/icon-192.png", "scope"]),
])
def test_broken_manifests_are_caught(name, failed):
    text = (HERE / "data" / "manifests" / f"{name}.json").read_text(encoding="utf-8")
    assert M.failed(M.check_manifest(text, "http://127.0.0.1:8000/", PUBLIC)) == failed


def test_wrong_size_passes_chromium_group_but_not_file_group():
    text = (HERE / "data" / "manifests" / "broken_wrong_size.json").read_text(encoding="utf-8")
    f = M.check_manifest(text, "http://127.0.0.1:8000/", PUBLIC)
    assert M.installable_in_chromium(f)
    assert any(x["group"] == "file" and not x["ok"] for x in f)


def test_secure_context_rule():
    assert M.secure_context("https://card.example/")
    assert M.secure_context("http://localhost:3000/")
    assert M.secure_context("http://127.0.0.1:8000/")
    assert not M.secure_context("http://card.example/")
    assert not M.secure_context("http://192.168.0.10:8000/")


def test_display_override_counts():
    m = {"name": "x", "start_url": "/", "display": "browser", "display_override": ["standalone"],
         "icons": [{"src": "a.png", "sizes": "192x192"}, {"src": "b.png", "sizes": "512x512"}]}
    assert M.installable_in_chromium(M.check_manifest(json.dumps(m)))


def test_icons_are_real_pngs_of_declared_size():
    for size in (192, 512):
        data = (PUBLIC / "icons" / f"icon-{size}.png").read_bytes()
        assert I.png_size(data) == (size, size)
        assert data == I.make_icon(size)  # 다시 만들어도 같은 바이트
    assert I.png_size(b"not a png") is None


# ---------- 서비스 워커 ----------

def test_sw_config_parses_and_precache_files_exist():
    cfg = C.load_sw_config()
    assert cfg["version"] == "card-editor-v1"
    assert len(cfg["precache"]) == 12
    for p in cfg["precache"]:
        assert (PUBLIC / ("index.html" if p == "/" else p.lstrip("/"))).is_file(), p
    assert set(C.PAGE_GETS) <= set(cfg["precache"])


def test_routes_pick_expected_strategy():
    cfg = C.load_sw_config()
    assert C.route_strategy(cfg, "/", True) == "network-first"
    assert C.route_strategy(cfg, "/templates/notice.json", False) == "stale-while-revalidate"
    assert C.route_strategy(cfg, "/api/generate-image", False) == "network-only"
    assert C.route_strategy(cfg, "/app.min.js", False) == "cache-first"


def test_sw_network_requests_skip_http_cache():
    js = read("sw.js")
    assert 'new Request(u, { cache: "reload" })' in js      # 설치 때 미리 받기
    assert 'return fetch(req, { cache: "reload" });' in js  # 실행 중 네트워크 요청
    assert js.count("fetch(req") == 1                         # 모든 네트워크 요청이 fromNetwork 를 거친다


def test_every_path_has_sources():
    assert all(p["sources"] for p in P.load_paths()["paths"])


def test_sw_skips_post_and_other_origins():
    js = read("sw.js")
    assert 'if (req.method !== "GET") return;' in js
    assert "if (new URL(req.url).origin !== self.location.origin) return;" in js


@pytest.mark.skipif(shutil.which("node") is None, reason="node 없음")
def test_js_syntax():
    for f in ("sw.js", "app.min.js"):
        assert subprocess.run(["node", "--check", str(PUBLIC / f)], capture_output=True).returncode == 0


def test_app_min_js_has_no_sourcemap_line():
    assert "sourceMappingURL" not in read("app.min.js")


def test_server_serves_disk_bytes_and_headers():
    with S.running() as base:
        for p in ["/", "/manifest.json", "/sw.js", "/icons/icon-512.png"]:
            r = urllib.request.urlopen(base + p)
            assert r.read() == (PUBLIC / ("index.html" if p == "/" else p.lstrip("/"))).read_bytes()
        assert urllib.request.urlopen(base + "/sw.js").headers["Cache-Control"] == "no-cache"
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(base + "/icons/")
        assert e.value.code == 404


# ---------- 캐시 전략 모형 ----------

def test_first_visit_hand_count():
    v = C.run("mixed")["visits"][0]
    assert v["network_requests"] == 8 + 1 + 12
    assert v["network_bytes"] == 7793 + 11405


def test_no_service_worker_fails_offline():
    r = C.run("none")
    assert [v["page_opened"] for v in r["visits"]] == [True, False, True, True, False]
    # HTTP 캐시(하루)에 남은 옛 notice.json, app.min.js 를 배포 뒤에도 준다
    assert r["visits"][2]["stale_paths"] == ["/app.min.js", "/templates/notice.json"]


def test_every_sw_strategy_opens_offline_but_post_fails():
    for s in ("cache-first", "network-first", "stale-while-revalidate", "mixed"):
        r = C.run(s)
        assert all(v["page_opened"] for v in r["visits"])
        assert [v["failed"] for v in r["visits"]] == [0, 1, 0, 0, 1]  # 오프라인 AI 버튼만 실패


def test_staleness_pattern_by_strategy():
    stale = lambda s, b=True: [v["stale"] for v in C.run(s, b)["visits"]]
    assert stale("network-first") == [0, 0, 0, 0, 0]
    assert stale("stale-while-revalidate") == [0, 0, 3, 0, 0]
    assert stale("cache-first") == [0, 0, 3, 0, 0]
    assert stale("cache-first", False) == [0, 0, 3, 3, 3]  # 버전을 안 바꾸면 계속 옛것
    assert stale("mixed") == [0, 0, 2, 0, 0]
    assert stale("mixed", False) == [0, 0, 2, 1, 1]  # cache-first 길의 app.min.js 만 남는다


def test_swr_does_not_save_requests_compared_to_network_first():
    a = C.run("stale-while-revalidate")["total"]
    b = C.run("network-first")["total"]
    assert a["network_requests"] == b["network_requests"]
    assert a["from_cache"] > b["from_cache"]


def test_post_cannot_be_cached_in_model():
    assert C.ACTION_POST not in C.load_sw_config()["precache"]


# ---------- 네 가지 길 ----------

def test_choose_has_no_ranking_and_keeps_table_order():
    data = P.load_paths()
    names = [p["name"] for p in data["paths"]]
    r = P.choose(["ios_bluetooth"], data)
    assert r["fits"] == [n for n in names if n in r["fits"]]
    assert r["fits"] == ["Capacitor", "React Native", "Flutter", "네이티브(Swift·Kotlin)"]


def test_cases():
    assert P.choose(P.CASES["A 링크로 퍼뜨리고, 새 템플릿 알림은 iOS에도"])["fits"] == ["PWA", "Capacitor"]
    assert P.choose(P.CASES["B 앱스토어 검색에 나오되 웹 코드는 그대로"])["fits"] == ["Capacitor"]
    assert P.choose(P.CASES["D 심사 없이 오늘 바로 휴대폰에 설치"])["fits"] == ["PWA"]
    assert P.choose(P.CASES["E 웹 코드 그대로 + iOS 블루투스 + 심사 없이"])["fits"] == []
    with pytest.raises(KeyError):
        P.choose(["fast"])


def test_every_path_source_id_exists():
    ids = {f["id"] for f in json.loads((HERE / "data" / "sources_2026-10-10.json").read_text(encoding="utf-8"))["facts"]}
    for p in P.load_paths()["paths"]:
        assert set(p["sources"]) <= ids


# ---------- 결과 파일 ----------

def test_results_file_matches():
    assert RESULTS["e3_first_visit"]["network_requests"] == 21
    assert RESULTS["e2_precache"]["total_bytes"] == sum(C.file_bytes(p) for p in C.load_sw_config()["precache"])
    for key, run in RESULTS["e4_strategies"]["runs"].items():
        s, b = key.split("|")
        assert C.run(s, b == "bump")["total"] == run["total"], key
    for name, case in RESULTS["e5_paths"]["cases"].items():
        assert P.choose(P.CASES[name]) == case
    assert RESULTS["e6_crosscheck"]["precache_missing"] == []
    assert RESULTS["e6_crosscheck"]["served_mismatch"] == []


def test_exercise_drop_one_condition():
    assert P.choose(["reuse_web_code", "ios_bluetooth"])["fits"] == ["Capacitor"]
    assert P.choose(["reuse_web_code", "link_install_no_review"])["fits"] == ["PWA"]
    assert P.choose(["ios_bluetooth", "link_install_no_review"])["fits"] == []


def test_exercise_precache_without_icons():
    cfg = C.load_sw_config()
    rest = [p for p in cfg["precache"] if not p.startswith("/icons/")]
    assert len(C.PAGE_GETS) + 1 + len(rest) == 19
    assert sum(C.file_bytes(p) for p in rest) == 8412
