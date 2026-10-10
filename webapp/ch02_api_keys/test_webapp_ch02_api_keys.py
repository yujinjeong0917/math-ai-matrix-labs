"""1인 개발 앱 출시 실전 2장 검증. `uv run pytest -q webapp/ch02_api_keys` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 네트워크와 진짜 키 없이 돈다.
진짜 접두사 규칙은 실행 중에 만든 문자열로만 시험하고, 그런 문자열을 파일에 남기지 않는다.
"""

import base64
import json
import re
from pathlib import Path

import pytest

import webapp_ch02_build as B
import webapp_ch02_rls as R
import webapp_ch02_scan as S

HERE = Path(__file__).parent
CH01 = HERE.parent / "ch01_everything_visible" / "editor"
RULES = S.load_rules()
REAL_PREFIXES = [r["prefix"] for r in RULES["rules"] if r["prefix"] and r["prefix"] != "eyJ"]


# ---------- 실습 폴더에 진짜 모양의 키가 없다 ----------

def test_no_real_key_shapes_on_disk():
    pat = re.compile(r"(?<![A-Za-z0-9_\-])(" + "|".join(map(re.escape, REAL_PREFIXES)) + r")[A-Za-z0-9_\-]{8,}")
    jwt = re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.")
    for p in HERE.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            text = p.read_text(encoding="utf-8", errors="replace")
            assert not pat.search(text), p
            assert not jwt.search(text), p


def test_fake_values_all_start_with_demo():
    for name in ("env.good", "env.mistake"):
        for k, v in B.read_env(B.PROJECT / name).items():
            if k.endswith("_KEY"):
                assert v.startswith("DEMO_"), (name, k)


# ---------- 키 구분표 ----------

def test_rules_have_source_and_kind():
    assert RULES["checked"] == "2026-10-10"
    kinds = [r["kind"] for r in RULES["rules"]]
    assert len(kinds) == 12 and kinds.count("public") == 3 and kinds.count("secret") == 7
    for r in RULES["rules"]:
        assert r["url"].startswith("https://")
        if r["prefix"] is None:
            assert "[확인 필요]" in r["prefix_note"]


# ---------- 진짜 접두사 규칙 (실행 중에 만든 문자열) ----------

@pytest.mark.parametrize("rule", [r for r in RULES["rules"] if r["prefix"] and r["prefix"] != "eyJ"],
                         ids=lambda r: r["id"])
def test_real_prefix_rules(rule):
    text = 'const k = "' + rule["prefix"] + "x" * 24 + '";'
    f = S.scan_text(text, "real", RULES)
    assert [(x["rule"], x["kind"]) for x in f] == [(rule["id"], rule["kind"])]
    # 접두사 뒤 6글자까지만 보이고 값 전체는 남기지 않아요
    assert f[0]["shown"] == rule["prefix"] + "xxxxxx…"
    assert "x" * 7 not in f[0]["shown"]


def test_real_prefix_needs_boundary_and_tail():
    assert S.scan_text('const risk_test_level = "aaaaaaaaaa";', "real", RULES) == []
    short = 'const k = "' + "sb_" + "secret_" + 'abc";'
    assert S.scan_text(short, "real", RULES) == []


def _b64(d):
    return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")


@pytest.mark.parametrize("role,kind", [("anon", "public"), ("service_role", "secret")])
def test_legacy_jwt_role(role, kind):
    tok = _b64({"alg": "HS256"}) + "." + _b64({"role": role}) + ".signature" + "y" * 10
    f = S.scan_text('key="' + tok + '"', "real", RULES)
    assert [(x["rule"], x["kind"]) for x in f] == [("supabase_legacy_jwt", kind)]


# ---------- 1장 편집기 ----------

def test_ch01_public_has_no_known_key_shapes():
    for mode in ("real", "demo"):
        rep = S.scan_dir(CH01 / "public", mode, RULES)
        assert rep["files"] == 9 and rep["findings"] == [] and rep["env_files"] == []


def test_ch01_server_only_key_found_only_by_name():
    rep = S.scan_dir(CH01 / "server_only", "real", RULES)
    assert [(f["file"], f["rule"], f["kind"]) for f in rep["findings"]] == [
        ("ai_config.json", "name:image_api_key", "check")]


# ---------- 빌드 두 판 ----------

def test_only_vite_names_reach_browser():
    good = B.build("good", write=False)
    assert good["inlined"] == ["VITE_SUPABASE_URL", "VITE_SUPABASE_PUBLISHABLE_KEY"]
    assert "DEMO_SUPABASE_SECRET_0002" not in good["out"] and "DEMO_AI_SECRET_0003" not in good["out"]
    bad = B.build("mistake", write=False)
    assert bad["inlined"] == ["VITE_SUPABASE_URL", "VITE_SUPABASE_SECRET_KEY", "VITE_AI_IMAGE_KEY"]
    assert "DEMO_SUPABASE_SECRET_0002" in bad["out"] and "DEMO_AI_SECRET_0003" in bad["out"]


def test_source_map_has_names_not_values():
    for w in B.BUILDS:
        b = B.build(w, write=False)
        assert "import.meta.env.VITE_" in b["map"]
        assert not any(v in b["map"] for v in b["env"].values() if v.startswith("DEMO_"))


def test_dist_is_fresh():
    for w in B.BUILDS:
        b = B.build(w, write=False)
        dist = B.PROJECT / B.BUILDS[w][2]
        assert (dist / "app.js").read_text(encoding="utf-8") == b["out"]
        assert (dist / "app.js.map").read_text(encoding="utf-8") == b["map"]


def test_scan_builds():
    good = S.summary(S.scan_dir(B.PROJECT / "dist_good", "demo", RULES))
    bad_rep = S.scan_dir(B.PROJECT / "dist_mistake", "demo", RULES)
    bad = S.summary(bad_rep)
    assert (good["public"], good["secret"], good["pass"]) == (1, 0, True)
    assert (bad["public"], bad["secret"], bad["pass"]) == (0, 2, False)
    assert sorted(f["rule"] for f in bad_rep["findings"]) == ["ai_api_key", "supabase_secret"]


def test_env_file_in_deploy_folder_fails(tmp_path):
    (tmp_path / "app.js").write_text("console.log(1)\n", encoding="utf-8")
    (tmp_path / ".env").write_text("NOTHING=1\n", encoding="utf-8")
    s = S.summary(S.scan_dir(tmp_path, "demo", RULES))
    assert s["env_files"] == 1 and s["pass"] is False


# ---------- RLS 모형 ----------

def test_rls_rows():
    got = {r["scenario"]: r["rows"] for r in R.run_scenarios()}
    assert list(got.values()) == [12, 0, 3, 6, 5, 7, 12, None]


def test_rls_sql_equals_python():
    for r in R.run_scenarios():
        if r["rows"] is not None:
            assert r["rows"] == r["python_rows"]


def test_policies_sql_matches_model():
    sql = (B.PROJECT / "supabase" / "policies.sql").read_text(encoding="utf-8")
    assert "enable row level security" in sql
    names = re.findall(r'create policy "([^"]+)"', sql)
    assert names == [p[0] for p in R.POLICIES]
    assert "to anon, authenticated" in sql and "(select auth.uid()) = owner_id" in sql


def test_results_file_matches():
    res = json.loads((HERE / "results" / "ch02.json").read_text(encoding="utf-8"))
    assert [r["rows"] for r in res["e3_rls"]] == [12, 0, 3, 6, 5, 7, 12, None]
    assert res["e2_builds"]["mistake"]["scan"]["demo"]["summary"]["secret"] == 2
    assert res["e4_real_rules_self_test"]["detected"] == res["e4_real_rules_self_test"]["samples"] == 11
    assert res["e5_crosscheck"] == {"dist_matches_build": True, "sql_equals_python": True}


def test_sources_file_has_url_and_date():
    src = json.loads((HERE / "data" / "sources_2026-10-10.json").read_text(encoding="utf-8"))
    assert src["checked"] == "2026-10-10"
    for f in src["facts"]:
        assert f["url"].startswith("https://") and f["checked"] == "2026-10-10"
