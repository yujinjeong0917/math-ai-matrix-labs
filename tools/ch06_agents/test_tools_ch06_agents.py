"""AI 도구 실전 6장 검증. `uv run pytest -q tools/ch06_agents` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
네트워크·API 키 없이 돈다.
"""

import json
import math
import tomllib
from fractions import Fraction as F
from pathlib import Path

import tools_ch06_chain as C
import tools_ch06_loop as L

HERE = Path(__file__).parent
FACTS = HERE / "data" / "agents_2026-10-10.json"
RES = json.loads((HERE / "results" / "ch06.json").read_text(encoding="utf-8"))


# ---------- 사실 파일과 설정 예시 ----------

def test_facts_have_url_and_date():
    d = json.loads(FACTS.read_text(encoding="utf-8"))
    assert d["checked"] == "2026-10-10"
    n = 0
    for prod in d["products"]:
        for fct in prod["facts"]:
            assert fct["url"].startswith("https://") and fct["checked"] == "2026-10-10", fct
            n += 1
    for fct in d["sources"]:
        assert fct["url"].startswith("https://") and fct["checked"] == "2026-10-10"
    assert n >= 30
    names = {p["name"] for p in d["products"]}
    assert {"Manus", "Claude Code", "OpenAI Codex"} <= names


def _fact(name, key):
    d = json.loads(FACTS.read_text(encoding="utf-8"))
    prod = next(p for p in d["products"] if p["name"] == name)
    return next(f["value"] for f in prod["facts"] if f["key"] == key)


def test_claude_settings_example_uses_documented_keys():
    cs = json.loads((HERE / "data" / "claude_settings.example.json").read_text(encoding="utf-8"))
    perm = cs["permissions"]
    assert set(perm) == {"defaultMode", "allow", "ask", "deny"}
    assert perm["defaultMode"] in [m.strip() for m in _fact("Claude Code", "modes").split(",")]
    assert "permissions.defaultMode" == _fact("Claude Code", "default_mode_key")
    assert any(r.startswith("Bash(git push") for r in perm["deny"])


def test_codex_config_example_uses_documented_values():
    cx = tomllib.loads((HERE / "data" / "codex_config.example.toml").read_text(encoding="utf-8"))
    sandbox_values = [v.strip() for v in _fact("OpenAI Codex", "sandbox_mode").split("|")]
    assert cx["sandbox_mode"] in sandbox_values and len(sandbox_values) == 3
    assert cx["approval_policy"] in ("on-request", "never")
    assert cx["approval_policy"] != "untrusted"          # 문서가 더는 지원하지 않는다고 적은 값
    assert cx["sandbox_workspace_write"]["network_access"] is False


# ---------- 닫힌 식 ----------

def test_hand_numbers():
    p = F(97, 100)
    assert C.chain(p, 2) == F(9409, 10000)
    assert C.chain(p, 3) == F(912673, 1000000)
    assert round(float(C.chain(p, 22)), 4) == 0.5117
    assert C.max_steps(p, F(9, 10)) == 3
    assert C.per_step_retry_perfect(p, 1) == F(9991, 10000)
    assert round(float(F(9991, 10000) ** 22), 4) == 0.9804


def test_project_has_22_steps_from_earlier_chapters():
    run = C.load_run()
    ids = [s["id"] for s in run["steps"]]
    assert len(ids) == 22
    assert sum(i.startswith("notion_") for i in ids) == 2       # 5장: 100행씩 2번
    assert sum(i.startswith("figma_") for i in ids) == 4        # 5장: Figma 4번
    assert sum(i.startswith("section_") for i in ids) == 12     # 2·3장: 절 12개
    assert [s["id"] for s in run["steps"] if s["undo"] == "hard"] == ["share"]


def test_step_outcomes_sum_to_one_and_reduce():
    for p in (F(9, 10), F(97, 100)):
        for d in (F(0), F(1, 2), F(1)):
            for k in range(4):
                s = C.step_outcomes(p, d, k)
                assert s["correct"] + s["missed"] + s["stopped"] == 1
    s = C.step_outcomes(F(97, 100), 0, 0)
    assert s["correct"] == F(97, 100) and s["stopped"] == 0
    s = C.step_outcomes(F(97, 100), 1, 1)
    assert s["correct"] == C.per_step_retry_perfect(F(97, 100), 1)


def test_run_outcomes_without_checks_is_p_to_the_n():
    o = C.run_outcomes(C.uniform(22, F(97, 100)))
    assert o["all_correct"] == F(97, 100) ** 22
    assert o["stopped_by_check"] == 0
    assert o["delivered_wrong"] == 1 - F(97, 100) ** 22        # 멈추지 않고 틀린 채 끝까지 간다


def test_first_failure_distribution_sums():
    p = F(97, 100)
    assert sum(C.first_failure_at(p, i) for i in range(1, 23)) + C.chain(p, 22) == 1


def test_break_even_share():
    r = C.agent_vs_workflow(F(97, 100), F(99, 100), 22, F(1, 10))
    e = r["break_even_odd_share"]
    wf = (1 - e) * r["workflow_normal"] + e * r["workflow_odd"]
    ag = (1 - e) * r["agent_normal"] + e * r["agent_odd"]
    assert wf == ag and 0 < e < 1


# ---------- 시뮬레이션과 결과 파일 ----------

def test_simulation_is_deterministic_and_within_3_se():
    a = C.assumptions()
    p, d, h = float(a["p"]), float(a["d"]), float(a["h"])
    s1 = L.simulate(L.run_workflow, 3000, p=p, d=d, k=a["k"], checks=True, human=True, h=h)
    s2 = L.simulate(L.run_workflow, 3000, p=p, d=d, k=a["k"], checks=True, human=True, h=h)
    assert s1 == s2
    for row in RES["e6_sim"]:
        if "z" in row:
            assert abs(row["z"]) < 3, row


def test_results_match_recomputation():
    a = C.assumptions()
    o = C.run_outcomes(C.uniform(22, a["p"], a["d"], a["k"]), human=a["h"], gate=21)
    sc = RES["e3_checks"]["scenarios"]["check_retry_human"]
    assert sc["all_correct"] == round(float(o["all_correct"]), 6)
    assert sc["delivered_wrong"] == round(float(o["delivered_wrong"]), 6)
    assert RES["e8_crosscheck"]["mismatches"] == 0
    assert RES["e0_hand"]["n22"] == 0.5117


def test_odd_week_workflow_with_check_stops_at_notion_2():
    row = next(r for r in RES["e6_sim"] if r["name"] == "workflow_check_odd")
    assert row["stop_steps"] == {"notion_2": row["weeks"]}
    row = next(r for r in RES["e6_sim"] if r["name"] == "workflow_none_odd")
    assert row["wrong"] == row["weeks"]


def test_agent_reads_third_page():
    import random
    rng = random.Random(0)
    run = C.load_run()
    log = L.run_agent(run["steps"], L.Week(230), rng, p=1.0, q=1.0)
    assert log["outcome"] == "correct" and log["attempts"] == 23
    log = L.run_workflow(run["steps"], L.Week(230), rng, p=1.0)
    assert log["outcome"] == "wrong"


def test_figma_quota_with_retries():
    so = C.step_outcomes(F(97, 100), F(9, 10), 2)
    month = 4 * so["attempts"] * F(52, 12)
    assert month < 20 and math.isclose(float(month), RES["e7_where"]["figma_calls_month_retry"], abs_tol=0.005)
