"""생산성 도구 6장 검증. `uv run pytest -q productivity/ch06_connect` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
"""

import copy
import json
import random
import statistics
from pathlib import Path

import productivity_ch06_graph as G
import productivity_ch06_registry as R
import productivity_ch06_sim as S

HERE = Path(__file__).parent


def test_first_screen_hand_numbers():
    # 복사: 세 곳을 손으로. 하나라도 틀릴 확률 1 - 0.9^3 = 0.271, 틀린 곳 수 기댓값 3 x 0.1 = 0.3
    assert round(1 - 0.9 ** 3, 3) == 0.271
    # 규칙: 컴포넌트 한 곳만 손으로. 하나라도 틀릴 확률 0.1, 놓치면 컴포넌트 + 인스턴스 2곳 = 3곳이 함께 틀림
    ew, pa = G.expected_mismatch(R.ITEMS["I01"], 0.9)
    assert round(pa, 4) == 0.1
    assert round(ew, 4) == 0.3
    assert 12 * 3 == 36 and 12 * 1 == 12  # 한 주 손 고치기 횟수


def test_registry_passes_and_injected_violations_are_found():
    assert G.validate(R.ITEMS) == []
    bad = copy.deepcopy(R.ITEMS)
    bad["I03"]["edges"].append(("sheets:metrics!threshold", "notion:Decisions/D04.기준", "auto"))  # 양방향
    found = G.validate(bad)
    assert any("고리" in msg for _, msg in found)
    bad2 = copy.deepcopy(R.ITEMS)
    bad2["I05"]["edges"].append(("sheets:lookup!price", "figma:S02_상세.운영시간", "human"))  # 두 곳에서 받음
    assert any("2곳에서 받음" in msg for _, msg in G.validate(bad2))


def test_three_reach_implementations_agree():
    assert G.correspondence(60, seed=11)["disagreements"] == 0
    edges = [e for it in R.ITEMS.values() for e in it["edges"]]
    reach = G.reach_bfs(edges, "figma:Components/취소안내")
    assert reach == {"figma:Components/취소안내", "figma:S02_상세.취소안내", "figma:S04_완료.취소안내"}
    assert G.reach_sql(edges, "figma:Components/취소안내") == reach


def test_graph_formula_matches_simulation():
    src_events = [e for e in R.EVENTS if e["where"] == R.ITEMS[e["item"]]["src"]]
    theory = S.expected_mismatch_from_source(src_events, 0.8)
    sims = [S.run_week("registry", 0.8, random.Random(s), events=src_events)["mismatch"] for s in range(3000)]
    se = statistics.stdev(sims) / len(sims) ** 0.5
    assert abs(statistics.mean(sims) - theory) < 4 * se


def test_permission_table_blocks_only_edits_without_rights():
    blocked = [e["id"] for e in R.EVENTS if e["where"] != R.ITEMS[e["item"]]["src"] and not R.can_edit(e["by"], e["where"])]
    assert blocked == ["E06", "E09"]  # 운영 담당이 Figma 화면 문구, 정리 시트를 바로 고친 경우
    # 디자이너가 Figma 컴포넌트에서 버튼 문구를 바꾼 E01은 권한이 막지 못한다(Figma 편집 권한이 정당하게 있음)
    assert R.can_edit("민지", "figma:Components/버튼")
    r = S.run_week("rules", 1.0, random.Random(0))
    assert r["blocked_by_permission"] == 2 and r["mismatch"] == 0  # q = 1이면 규칙 방식은 어긋남 0


def test_documented_automation_traps():
    # 이름으로 찾는 스크립트는 이름이 바뀐 날부터 멈추고, id로 찾으면 만든 사람이 빠질 때까지 돈다
    assert S.automation_alive("A3", S.RENAME_DAY - 1, "auto_naive")
    assert not S.automation_alive("A3", S.RENAME_DAY, "auto_naive")
    assert S.automation_alive("A3", S.RENAME_DAY, "auto_guarded")
    assert not S.automation_alive("A3", S.CREATOR_LEAVES_DAY, "auto_guarded")
    naive = S.run_week("auto_naive", 1.0, random.Random(0))
    assert ("I03", "sheets:metrics!threshold") in naive["wrong"]  # 사람이 다 맞게 해도 자동화 쪽은 틀림
    assert naive["dead_automation_unknown"] == 2
    guarded = S.run_week("auto_guarded", 1.0, random.Random(0))
    assert guarded["mismatch"] == 0 and guarded["checks_false"] == 2  # 마지막 실행 점검 칸 두 개가 FALSE로 드러남


def test_webhook_order_matches_formula_and_refetch_is_safe():
    for g in (10, 60, 180):
        r = S.webhook_order(g, n=40000, seed=3)
        assert abs(r["stale_rate_arrival_order"] - r["theory"]) < 0.01
        assert r["stale_rate_refetch"] == 0.0


def test_two_way_sync_loses_edits_and_one_way_does_not():
    two = S.two_way_sync(5, days=300, seed=1)
    one = S.two_way_sync(5, days=300, seed=1, one_way=True)
    assert two["clobbered_edits"] > 0 and two["days_diverged"] > 0
    assert one["clobbered_edits"] == 0 and one["days_diverged"] == 0


def test_results_file_matches_recomputation():
    res = json.loads((HERE / "results" / "ch06.json").read_text())
    assert res["e0_hand"]["copy_p_any_wrong"] == 0.271
    assert res["e1_validate"]["base_problems"] == []
    assert res["e8_correspondence"]["disagreements"] == 0
    m = S.summarize("rules", 0.9, 1000)
    assert m["mismatch"] == res["e2_week"]["means"]["rules"]["mismatch"]
