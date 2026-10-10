"""생산성 도구 6장 비교 실험. `OMP_NUM_THREADS=2 uv run python productivity/ch06_connect/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 취소 규정 문구 하나가 네 곳에. 손으로 옮기는 곳 3곳(복사) vs 1곳(컴포넌트·링크)
E1 등록부 검사: 지금 등록부 통과 여부, 일부러 넣은 위반 세 가지(양방향 동기화, 원본 둘, 끊긴 위치)를 잡나
E2 1주 시나리오: 변경 12개 x 방식 5개 x 시드 1000개, q = 0.9
E3 q 바꿔 보기: 0.7 / 0.8 / 0.9 / 0.95
E4 그래프 식과 시뮬레이션 대조: 원본에서 시작한 이벤트만 모아 기댓값을 바로 계산
E5 자동화가 깨진 뒤 알아채기까지 걸린 날(시드 1000개)
E6 양방향 동기화: 지연 1 / 5 / 30분에서 옛 값에 덮인 편집, 하루 끝에 두 칸이 다른 날
E7 웹훅 도착 순서: 간격 10초 / 1분 / 3분 / 10분, 도착 순서로 적용 vs 다시 읽기
E8 대응 구현 대조: 무작위 등록부 500개에서 도달 집합 BFS / NumPy / sqlite3
E9 자동화 손익: 만드는 시간 vs 한 주에 아끼는 시간
E10 넘겨준 기록 2건: rules 방식 시드 0의 주말 상태에 3장 체크리스트와 5장 checks를 그대로 대 본 결과

고정 시드라 결과가 매번 같다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import copy  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import productivity_ch06_graph as G  # noqa: E402
import productivity_ch06_registry as R  # noqa: E402
import productivity_ch06_sim as S  # noqa: E402

OUT = HERE / "results" / "ch06.json"
SEEDS = 1000
Q = 0.9


def e0_hand():
    h = R.HAND
    m = h["miss"]
    q = 1 - m
    copy_any = 1 - q ** h["copy_human"]
    rule_any = 1 - q ** h["rule_human"]
    # 복사: 세 곳이 따로 틀린다. 틀린 곳 수 기댓값 = 3 x 0.1
    copy_exp = h["copy_human"] * m
    # 규칙: 컴포넌트를 놓치면 컴포넌트 + 인스턴스 두 화면 = 3곳이 한꺼번에 틀린다. 시트는 링크라 안 틀린다.
    rule_exp = m * 3
    # 같은 값을 등록부 그래프 식으로 다시 계산
    ew, pa = G.expected_mismatch(R.ITEMS["I01"], q)
    return {
        "places": h["places"], "miss": m, "changes": h["changes"],
        "copy_hand_places": h["copy_human"], "rule_hand_places": h["rule_human"],
        "copy_p_any_wrong": round(copy_any, 4), "rule_p_any_wrong": round(rule_any, 4),
        "copy_expected_wrong_places": round(copy_exp, 4), "rule_expected_wrong_places": round(rule_exp, 4),
        "graph_formula_rule": {"expected_wrong": round(ew, 4), "p_any": round(pa, 4)},
        "copy_hand_edits_per_week": h["copy_human"] * h["changes"],
        "rule_hand_edits_per_week": h["rule_human"] * h["changes"],
        "copy_expected_changes_with_miss": round(h["changes"] * copy_any, 3),
        "rule_expected_changes_with_miss": round(h["changes"] * rule_any, 3),
    }


def e1_validate():
    base = G.validate(R.ITEMS)
    bad = copy.deepcopy(R.ITEMS)
    # 위반 1: Sheets 기준 칸에서도 Notion 기준값을 고치게 양방향으로 잇는다
    bad["I03"]["edges"].append(("sheets:metrics!threshold", "notion:Decisions/D04.기준", "auto"))
    # 위반 2: 운영 시간을 Sheets에서도 따로 받는다(값을 두 곳에서 받음)
    bad["I05"]["edges"].append(("sheets:lookup!price", "figma:S02_상세.운영시간", "human"))
    # 위반 3: 지점 수 칸이 원본과 끊긴 채 따로 있다
    bad["I04"]["edges"] = [e for e in bad["I04"]["edges"] if e[1] != "notion:Projects.지점수"] + [
        ("notion:Projects.지점수", "notion:Projects.주간리포트지점", "human")]
    found = G.validate(bad)
    counts = R.count_methods()
    hops = {k: G.human_hops(it) for k, it in R.ITEMS.items()}
    max_h = max(v["human"] for h in hops.values() for v in h.values())
    return {"base_problems": base, "injected": 3, "found": [f"{k}: {msg}" for k, msg in found],
            "n_items": len(R.ITEMS), "n_locations": sum(len(R.locations(k)) for k in R.ITEMS),
            "edges": counts, "max_human_hops": max_h,
            "tools_edges": tool_pairs()}


def tool_pairs():
    """도구 사이 간선이 몇 개인지(같은 도구 안 간선은 제외)."""
    out = {}
    for _, a, b, m in R.all_edges():
        ta, tb = a.split(":")[0], b.split(":")[0]
        if ta != tb:
            key = f"{ta}->{tb}"
            out.setdefault(key, {"ref": 0, "human": 0, "auto": 0})
            out[key][m] += 1
    return out


def e2_week(q=Q):
    res = {reg: S.summarize(reg, q, SEEDS) for reg in S.REGIMES}
    # 한 시드에서 어디가 틀렸는지 예시(rules)
    ex = {}
    for reg in S.REGIMES:
        r = S.run_week(reg, q, random.Random(0))
        ex[reg] = sorted(f"{k} {loc}" for k, loc in r["wrong"])
    # 어떤 항목이 가장 자주 틀리나(rules, auto_naive)
    freq = {}
    for reg in ("none", "rules", "auto_naive", "auto_guarded"):
        c = {}
        for s in range(SEEDS):
            for k, loc in S.run_week(reg, q, random.Random(s))["wrong"]:
                c[f"{k} {R.ITEMS[k]['name']} @ {loc}"] = c.get(f"{k} {R.ITEMS[k]['name']} @ {loc}", 0) + 1
        freq[reg] = sorted(({"where": w, "rate": round(n / SEEDS, 3)} for w, n in c.items()),
                           key=lambda d: -d["rate"])[:6]
    return {"q": q, "seeds": SEEDS, "means": res, "seed0_wrong": ex, "most_often_wrong": freq}


def e3_sweep():
    out = {}
    for q in (0.7, 0.8, 0.9, 0.95):
        out[str(q)] = {reg: {k: v for k, v in S.summarize(reg, q, SEEDS).items()
                             if k in ("mismatch", "checklist_unmet", "hidden_mismatch", "untraceable", "minutes",
                                      "p_any_mismatch")}
                       for reg in S.REGIMES}
    return out


def e4_formula():
    src_events = [e for e in R.EVENTS if e["where"] == R.ITEMS[e["item"]]["src"]]
    out = []
    for q in (0.7, 0.9):
        theory = S.expected_mismatch_from_source(src_events, q)
        sims = [S.run_week("registry", q, random.Random(s), events=src_events)["mismatch"] for s in range(4000)]
        out.append({"q": q, "events": len(src_events), "runs": len(sims), "theory": round(theory, 4),
                    "sim_mean": round(statistics.mean(sims), 4),
                    "sim_se": round(statistics.stdev(sims) / len(sims) ** 0.5, 4)})
    return out


def e5_detection():
    out = {}
    for kind in ("email_creator_here", "email_creator_gone", "silent", "heartbeat"):
        rng = random.Random(5)
        days = [S.detection_days(kind, rng) for _ in range(SEEDS)]
        days.sort()
        out[kind] = {"mean_days": round(statistics.mean(days), 2), "median": days[len(days) // 2],
                     "p90": days[int(0.9 * len(days))], "still_unknown_after_14d": round(sum(d > 14 for d in days) / SEEDS, 3)}
    out["assumption_p_notice_per_day"] = 0.1
    return out


def e6_two_way():
    out = []
    for d in (1, 5, 30):
        two = S.two_way_sync(d)
        one = S.two_way_sync(d, one_way=True)
        out.append({"delay_min": d, "hours_per_day": 8, "two_way": two, "one_way": {"edits": one["edits"],
                    "clobbered_edits": one["clobbered_edits"], "days_diverged": one["days_diverged"]}})
    return out


def e7_webhook():
    return [S.webhook_order(g) for g in (10, 60, 180, 600)]


def e9_breakeven(week):
    m = week["means"]
    saved = round(m["rules"]["minutes"] - m["auto_naive"]["minutes"], 2)  # 자동화가 깨지지 않을 때 아끼는 시간
    setup = m["auto_guarded"]["setup_minutes"]
    fix = round(m["auto_guarded"]["minutes"] - m["auto_naive"]["minutes"], 2)
    return {"setup_minutes": setup, "saved_minutes_per_week": saved,
            "weeks_to_break_even_no_breaks": round(setup / saved, 1) if saved > 0 else None,
            "fix_minutes_this_week_guarded": fix,
            "note": "saved = rules 방식 시간 - 자동화 방식 시간(자동화가 대신한 손 옮기기). 시간 값은 모두 가정."}


def e10_handoff(seed=0, q=Q):
    r = S.run_week("rules", q, random.Random(seed))
    wrong = set(r["wrong"])
    # 디자인 -> 개발: Figma와 Notion 할 일로 가는 간선의 받는 곳을 하나씩 대조(3장 체크리스트의 문구·링크 줄)
    rows = []
    for k, a, b, m in R.all_edges():
        if S.CHECKLIST(b):
            rows.append({"item": k, "name": R.ITEMS[k]["name"], "location": b, "method": m,
                         "ok": (k, b) not in wrong})
    named = [rec for rec in r["log"] if rec["id"] == "E11"][0].get("named_version", False)
    design_dev = {"date": "7일째", "from": "민지", "to": "준호", "figma_named_version_before_handoff": named,
                  "rows_checked": len(rows), "rows_ok": sum(x["ok"] for x in rows),
                  "failed": [x for x in rows if not x["ok"]], "rows": rows}
    ops_plan = {"date": "7일째", "from": "하린", "to": "서연",
                "checks": {k: bool(v) for k, v in r["checks"].items()},
                "change_log_rows": sum(rec["logged"] for rec in r["log"]), "changes": len(r["log"]),
                "untraceable": r["untraceable"],
                "mismatch_not_seen_by_any_check": sorted(f"{k} {loc}" for k, loc in r["wrong"]
                                                         if not S.CHECKLIST(loc) and (k, loc) not in S.SHEET_CHECKS.values())}
    return {"seed": seed, "q": q, "design_to_dev": design_dev, "ops_to_planning": ops_plan}


def write_data():
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "registry.json").write_text(json.dumps(R.ITEMS, ensure_ascii=False, indent=1))
    (HERE / "data" / "events.json").write_text(json.dumps(
        {"events": R.EVENTS, "incidents": R.WEEK_INCIDENTS, "permissions": R.PERMISSIONS,
         "automations": R.AUTOMATIONS}, ensure_ascii=False, indent=1))


def write_handoff(h):
    (HERE / "data" / "handoff_records.json").write_text(json.dumps(h, ensure_ascii=False, indent=1))


def write_checks(week, h=None):
    (HERE / "checks").mkdir(exist_ok=True)
    m = week["means"]
    with open(HERE / "checks" / "ch06.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reader", "date", "method", "mismatch", "checklist_unmet", "checks_false", "untraceable",
                    "minutes", "note"])
        for reg in S.REGIMES:
            r = m[reg]
            w.writerow(["example", "2026-10-08", reg, r["mismatch"], r["checklist_unmet"], r["checks_false"],
                        r["untraceable"], r["minutes"], "합성 모형 평균(시드 1000개, q=0.9). 독자는 자기 기록을 한 줄씩 더한다"])
        if h:
            d, o = h["design_to_dev"], h["ops_to_planning"]
            w.writerow(["handoff", d["date"], "rules 디자인->개발", len(d["failed"]), len(d["failed"]), "", "", "",
                        f"대조 {d['rows_checked']}줄 중 {d['rows_ok']}줄 맞음, Figma 버전 이름 {'있음' if d['figma_named_version_before_handoff'] else '없음'}"])
            w.writerow(["handoff", o["date"], "rules 운영->기획", "", "", sum(not v for v in o["checks"].values()),
                        o["untraceable"], "", f"변경 기록 {o['change_log_rows']}/{o['changes']}줄, 점검 밖 불일치 {len(o['mismatch_not_seen_by_any_check'])}곳"])


def main():
    write_data()
    week = e2_week()
    hand = e10_handoff()
    write_handoff(hand)
    write_checks(week, hand)
    res = {
        "meta": {"python": platform.python_version(), "numpy": np.__version__,
                 "platform": f"{platform.system()} {platform.machine()}", "seeds": SEEDS, "q": Q,
                 "checked": "2026-10-08", "events": len(R.EVENTS), "items": len(R.ITEMS),
                 "time_assumptions_min": S.TIME, "p_note": S.P_NOTE, "review_day": S.REVIEW_DAY,
                 "history_days_free": S.HISTORY_DAYS},
        "e0_hand": e0_hand(),
        "e1_validate": e1_validate(),
        "e2_week": week,
        "e3_sweep": e3_sweep(),
        "e4_formula": e4_formula(),
        "e5_detection": e5_detection(),
        "e6_two_way": e6_two_way(),
        "e7_webhook": e7_webhook(),
        "e8_correspondence": G.correspondence(500),
        "e9_breakeven": e9_breakeven(week),
        "e10_handoff": hand,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
