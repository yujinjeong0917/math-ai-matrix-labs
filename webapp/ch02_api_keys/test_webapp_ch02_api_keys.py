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
# eyJ(JWT)는 접두사 규칙이 아니라 별도 정규식으로 다루므로 뺀다.
REAL_PREFIXES = [r["prefix"] for r in RULES["rules"] if r["prefix"] and r["prefix"] != "eyJ"]


# ---------- 실습 폴더에 진짜 모양의 키가 없다 ----------

def test_no_real_key_shapes_on_disk():
    # 실습 폴더 전체(데이터·결과 포함)에 진짜 접두사 모양 + 8글자 이상이 없는지 훑는다. 가짜 값은 모두 DEMO_ 표식으로만 만든다.
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
    # 규칙 12개 = public 3 + secret 7 + by_role 1(옛 JWT) + depends 1(Google 키). 접두사가 없는 규칙은 "[확인 필요]"를 달아 근거 강도를 밝힌다.
    assert len(kinds) == 12 and kinds.count("public") == 3 and kinds.count("secret") == 7
    for r in RULES["rules"]:
        assert r["url"].startswith("https://")
        if r["prefix"] is None:
            assert "[확인 필요]" in r["prefix_note"]


# ---------- 진짜 접두사 규칙 (실행 중에 만든 문자열) ----------

@pytest.mark.parametrize("rule", [r for r in RULES["rules"] if r["prefix"] and r["prefix"] != "eyJ"],
                         ids=lambda r: r["id"])
def test_real_prefix_rules(rule):
    # 접두사 + x 24개를 실행 중에 만들어 시험. 이 문자열을 파일에 쓰지 않으므로 위의 디스크 검사와 충돌하지 않는다.
    text = 'const k = "' + rule["prefix"] + "x" * 24 + '";'
    f = S.scan_text(text, "real", RULES)
    assert [(x["rule"], x["kind"]) for x in f] == [(rule["id"], rule["kind"])]
    # 접두사 뒤 6글자까지만 보이고 값 전체는 남기지 않아요
    # 마스킹 규칙: 접두사 뒤 6글자만 보이고 나머지는 … 로 대체.
    assert f[0]["shown"] == rule["prefix"] + "xxxxxx…"
    assert "x" * 7 not in f[0]["shown"]


def test_real_prefix_needs_boundary_and_tail():
    # 경계 시험: 'risk_test_level'처럼 단어 중간에 접두사 글자가 우연히 들어간 경우는 찾지 않아야 한다.
    assert S.scan_text('const risk_test_level = "aaaaaaaaaa";', "real", RULES) == []
    # 접두사를 문자열 덧셈으로 쪼개 쓴 것은 소스 파일에 접두사 모양이 남지 않게 하려는 것. 뒤가 3글자(abc)뿐이라 TAIL 8글자 미만이라 무시.
    short = 'const k = "' + "sb_" + "secret_" + 'abc";'
    assert S.scan_text(short, "real", RULES) == []


def _b64(d):
    return base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")


@pytest.mark.parametrize("role,kind", [("anon", "public"), ("service_role", "secret")])
def test_legacy_jwt_role(role, kind):
    # 머리.내용.서명 세 토막의 가짜 JWT. 가운데 토막의 role만 판정에 쓰인다.
    tok = _b64({"alg": "HS256"}) + "." + _b64({"role": role}) + ".signature" + "y" * 10
    f = S.scan_text('key="' + tok + '"', "real", RULES)
    assert [(x["rule"], x["kind"]) for x in f] == [("supabase_legacy_jwt", kind)]


# ---------- 1장 편집기 ----------

def test_ch01_public_has_no_known_key_shapes():
    for mode in ("real", "demo"):
        rep = S.scan_dir(CH01 / "public", mode, RULES)
        # 1장 공개 폴더의 글자 파일 9개를 훑어 키 모양 0건, .env 0개.
        assert rep["files"] == 9 and rep["findings"] == [] and rep["env_files"] == []


def test_ch01_server_only_key_found_only_by_name():
    rep = S.scan_dir(CH01 / "server_only", "real", RULES)
    # 서버 폴더의 가짜 키는 모양 규칙으로는 안 잡히고 이름 규칙(image_api_key)으로만 잡혀 "확인 필요"로 분류된다.
    assert [(f["file"], f["rule"], f["kind"]) for f in rep["findings"]] == [
        ("ai_config.json", "name:image_api_key", "check")]


# ---------- 빌드 두 판 ----------

def test_only_vite_names_reach_browser():
    good = B.build("good", write=False)
    # VITE_ 접두사 이름만 브라우저 번들에 들어간다. 비밀 값(DEMO_..._SECRET)은 good 빌드에 없고 mistake 빌드에 있다.
    assert good["inlined"] == ["VITE_SUPABASE_URL", "VITE_SUPABASE_PUBLISHABLE_KEY"]
    assert "DEMO_SUPABASE_SECRET_0002" not in good["out"] and "DEMO_AI_SECRET_0003" not in good["out"]
    bad = B.build("mistake", write=False)
    assert bad["inlined"] == ["VITE_SUPABASE_URL", "VITE_SUPABASE_SECRET_KEY", "VITE_AI_IMAGE_KEY"]
    assert "DEMO_SUPABASE_SECRET_0002" in bad["out"] and "DEMO_AI_SECRET_0003" in bad["out"]


def test_source_map_has_names_not_values():
    for w in B.BUILDS:
        b = B.build(w, write=False)
        assert "import.meta.env.VITE_" in b["map"]
        # 지도 파일에는 원본 소스가 들어 있어 import.meta.env.VITE_ 이름은 보이지만, 값은 소스에 쓰여 있지 않아 지도에도 없다.
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
    # good: 공개 키 1건, 비밀 0건 -> 통과. mistake: 비밀 2건(supabase_secret, ai_api_key) -> 실패. demo 표식 기준.
    assert (good["public"], good["secret"], good["pass"]) == (1, 0, True)
    assert (bad["public"], bad["secret"], bad["pass"]) == (0, 2, False)
    assert sorted(f["rule"] for f in bad_rep["findings"]) == ["ai_api_key", "supabase_secret"]


def test_env_file_in_deploy_folder_fails(tmp_path):
    (tmp_path / "app.js").write_text("console.log(1)\n", encoding="utf-8")
    # 키 값이 없어도 .env 파일이 배포 폴더에 있으면 실패해야 한다.
    (tmp_path / ".env").write_text("NOTHING=1\n", encoding="utf-8")
    s = S.summary(S.scan_dir(tmp_path, "demo", RULES))
    assert s["env_files"] == 1 and s["pass"] is False


# ---------- RLS 모형 ----------

def test_rls_rows():
    got = {r["scenario"]: r["rows"] for r in R.run_scenarios()}
    # 카드 12장(alice 5, bob 3, carol 4; 공개 2+1+0 = 3장). 기대값: 꺼짐 12, 정책 없음 0, 익명 3, alice 5+1 = 6, bob 3+2 = 5, carol 4+3 = 7, 비밀 키 12, grant 없음 None(권한 오류).
    assert list(got.values()) == [12, 0, 3, 6, 5, 7, 12, None]


def test_rls_sql_equals_python():
    for r in R.run_scenarios():
        if r["rows"] is not None:
            assert r["rows"] == r["python_rows"]


def test_policies_sql_matches_model():
    sql = (B.PROJECT / "supabase" / "policies.sql").read_text(encoding="utf-8")
    # policies.sql 문자열의 일부를 확인하는 간단한 대조. (select auth.uid())로 감싸는 형태는 Supabase 문서가 성능상 권하는 작성법.
    assert "enable row level security" in sql
    names = re.findall(r'create policy "([^"]+)"', sql)
    assert names == [p[0] for p in R.POLICIES]
    assert "to anon, authenticated" in sql and "(select auth.uid()) = owner_id" in sql


def test_results_file_matches():
    res = json.loads((HERE / "results" / "ch02.json").read_text(encoding="utf-8"))
    assert [r["rows"] for r in res["e3_rls"]] == [12, 0, 3, 6, 5, 7, 12, None]
    assert res["e2_builds"]["mistake"]["scan"]["demo"]["summary"]["secret"] == 2
    # 11 = 접두사 규칙 9개(Supabase 2 + Stripe 7) + JWT 2종(anon, service_role).
    assert res["e4_real_rules_self_test"]["detected"] == res["e4_real_rules_self_test"]["samples"] == 11
    assert res["e5_crosscheck"] == {"dist_matches_build": True, "sql_equals_python": True}


def test_sources_file_has_url_and_date():
    src = json.loads((HERE / "data" / "sources_2026-10-10.json").read_text(encoding="utf-8"))
    assert src["checked"] == "2026-10-10"
    for f in src["facts"]:
        assert f["url"].startswith("https://") and f["checked"] == "2026-10-10"
