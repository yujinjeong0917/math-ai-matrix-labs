"""1인 개발 앱 출시 실전 1장 검증. `uv run pytest -q webapp/ch01_everything_visible` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
서버는 127.0.0.1의 빈 포트에 잠깐 띄운다. 바깥 네트워크와 API 키 없이 돈다.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

import webapp_ch01_build as B
import webapp_ch01_server as S
import webapp_ch01_visible as V

HERE = Path(__file__).parent
FAKE_KEY = "DEMO_NOT_A_REAL_KEY_0000"


@pytest.fixture(scope="module")
def visit_log():
    with V.running_editor(serve_maps=True) as base:
        yield V.visit(base)


# ---------- 처음 열 때 무엇이 내려오나 ----------

def test_first_open_downloads_eight_files_of_every_kind(visit_log):
    first = [r for r in visit_log if r["step"] == "open"]
    assert [r["path"] for r in first] == [
        "/", "/images/logo.svg", "/style.css", "/fonts/card-sans.woff2", "/images/bg-dots.svg",
        "/app.min.js", "/templates/index.json", "/templates/notice.json"]
    assert {r["kind"] for r in first} == {"document", "img", "css", "font", "js", "json"}
    assert all(r["status"] == 200 for r in first)
    # 첫 화면 손계산
    assert [r["bytes"] for r in first] == [1079, 360, 1027, 84, 3199, 1487, 168, 255]
    assert sum(r["bytes"] for r in first) == 7659


def test_actions_add_template_and_api(visit_log):
    later = [(r["step"], r["method"], r["path"]) for r in visit_log if r["step"] != "open"]
    assert later == [("change_template", "GET", "/templates/quote.json"),
                     ("make_background", "POST", "/api/generate-image")]


def test_sizes_match_disk(visit_log):
    for r in visit_log:
        if r["method"] == "GET":
            rel = "index.html" if r["path"] == "/" else r["path"].lstrip("/")
            assert (S.PUBLIC / rel).stat().st_size == r["bytes"]


# ---------- 서버에만 남은 것 ----------

def test_server_only_values_never_downloaded(visit_log):
    blob = V.all_downloaded_bytes(visit_log)
    assert FAKE_KEY.encode() not in blob
    assert "글자를 넣지 말 것".encode() not in blob
    assert "작업 원본".encode() not in blob
    # API 응답(생성된 그림)은 내려가지만 키는 없다
    api = [r for r in visit_log if r["kind"] == "api"][0]
    assert api["status"] == 200 and api["_body"].startswith(b"<svg") and FAKE_KEY.encode() not in api["_body"]


def test_fake_key_is_obviously_fake():
    cfg = S.load_server_config()
    assert cfg["image_api_key"] == FAKE_KEY
    assert not cfg["image_api_key"].startswith(("sk-", "AIza", "ghp_", "xox"))


def test_inventory_split(visit_log):
    inv = {r["file"]: r["where"] for r in V.inventory(visit_log)}
    assert inv["server_only/ai_config.json"] == "서버에만 있음"
    assert inv["server_only/templates_master/notice_master_1080.svg"] == "서버에만 있음"
    assert inv["src/app.js"] == "서버에만 있음"
    assert inv["public/app.min.js.map"].startswith("공개 폴더")
    assert sum(1 for w in inv.values() if w == "내려감") == 9


def test_outside_public_and_listing_are_404():
    with V.running_editor() as base:
        for p in ["/server_only/ai_config.json", "/src/app.js", "/templates/", "/images/"]:
            assert V.request(base, p)[0] == 404


def test_model_refuses_other_hosts():
    with pytest.raises(ValueError):
        V.request("http://example.com", "/")


# ---------- 압축과 지도 파일 ----------

def test_minify_removes_comments_but_keeps_strings():
    original, minified, _ = B.build(write=False)
    assert len(B.comments(original)) == 4
    assert B.comments(minified) == ["//# sourceMappingURL=app.min.js.map"]
    assert '"/api/generate-image"' in minified and '"templates/index.json"' in minified
    assert "currentTemplate" not in minified and "GENERATE_ENDPOINT" not in minified
    assert len(minified.encode()) == 1487 and len(original.encode()) == 2715


def test_public_build_is_fresh():
    _, minified, smap = B.build(write=False)
    assert (B.PUBLIC / B.MIN_NAME).read_text(encoding="utf-8") == minified
    assert (B.PUBLIC / B.MAP_NAME).read_text(encoding="utf-8") == smap


def test_minify_handles_slashes_inside_strings():
    src = 'const u = "a//b"; // 주석\nconst v = "/* 아님 */";\n'
    out = B.minify(src, rename=False)
    assert '"a//b"' in out and '"/* 아님 */"' in out and "주석" not in out


@pytest.mark.skipif(shutil.which("node") is None, reason="node 없음")
def test_minified_js_parses_with_node():
    r = subprocess.run(["node", "--check", str(B.PUBLIC / B.MIN_NAME)], capture_output=True)
    assert r.returncode == 0, r.stderr


def test_source_map_brings_back_original_when_served():
    original = B.SRC.read_text(encoding="utf-8")
    with V.running_editor(serve_maps=True) as base:
        log = V.visit(base, actions=(), devtools_source_maps=True)
    m = [r for r in log if r["kind"] == "sourcemap"][0]
    assert m["status"] == 200
    assert json.loads(m["_body"])["sourcesContent"][0] == original
    with V.running_editor(serve_maps=False) as base:
        log = V.visit(base, actions=(), devtools_source_maps=True)
    assert [r for r in log if r["kind"] == "sourcemap"][0]["status"] == 404


# ---------- 캐시 ----------

def test_cache_rules(visit_log):
    c = V.cache_after_first_visit(visit_log)
    assert c["not_stored"] == ["/api/generate-image"]
    assert c["ask_server_again"] == ["/"]
    assert len(c["stored"]) == 9 and c["stored_bytes"] == 7914


def test_results_file_matches():
    res = json.loads((HERE / "results" / "ch01.json").read_text(encoding="utf-8"))
    assert res["e0_first_open"]["open_files"] == 8 and res["e0_first_open"]["open_bytes"] == 7659
    assert all(v == 0 for v in res["e3_server_only_markers_in_downloads"].values())
    assert res["e8_crosscheck"]["mismatches"] == 0
    assert res["e2_summary"]["server_only_bytes"] == 219 + 311 + 42640 + 2715


def test_sources_file_has_url_and_date():
    src = json.loads((HERE / "data" / "sources_2026-10-10.json").read_text(encoding="utf-8"))
    assert src["checked"] == "2026-10-10"
    for f in src["facts"]:
        assert f["url"].startswith("https://") and f["checked"] == "2026-10-10"
