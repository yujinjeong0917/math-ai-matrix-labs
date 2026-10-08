"""생산성 도구 4장 비교 실험. `uv run python productivity/ch04_notion_relations/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 결정 3개 × 할 일 5개 표에서 결정별 미완료 할 일 수
E1 실패 최소 예제: 일반 페이지(관계 없음)로 질문 5개, 모든 연결을 적은 경우(r = 1.0)
E2 기록률 r을 낮출 때: 페이지 / 관계형 / 관계형 + 빈칸 검사 후 다시 채움 (같은 시드에서 같은 연결을 잃음)
E3 빈칸 검사가 잃은 연결을 얼마나 잡나(종류별)
E4 연결 함정(Codd 1970): S03에 영향을 준 결정을 할 일 경로로 고를 때와 직접 relation으로 고를 때
E5 결정 번복: D07 → D10, 영향받는 할 일과 옛 결정 페이지에서 번복이 보이는지
E6 손으로 적은 개수 vs rollup: 2주 동안 60번 바뀔 때 어긋나는 비율
E7 대응 구현 대조: 무작위 DB 500개에서 순수 파이썬 / sqlite3 / 수식 모양 rollup

외부 패키지 없이 표준 라이브러리만 쓴다. 고정 시드라 결과는 매번 같다.
"""

import json
import platform
import random
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import productivity_ch04_data as DATA  # noqa: E402
import productivity_ch04_pages as P  # noqa: E402
import productivity_ch04_relational as R  # noqa: E402

OUT = HERE / "results" / "ch04.json"
SEED = 4
N_SEEDS = 200
RATES = [1.0, 0.9, 0.8, 0.6]
QS = ["Q1", "Q2", "Q3", "Q4", "Q5"]


def truth():
    return R.answer_all(R.build())


def grade(ans, tru):
    if ans == P.UNANSWERABLE:
        return "unanswerable"
    return "correct" if ans == tru else "wrong"


def as_graded(a):
    """페이지 방식과 같은 규칙: Q1에 할 일 0개, Q4에 빈칸이 있으면 답할 수 없음."""
    a = dict(a)
    if not a["Q1"]:
        a["Q1"] = P.UNANSWERABLE
    if any(v is None for v in a["Q4"].values()):
        a["Q4"] = P.UNANSWERABLE
    return a


def lose(rng, r):
    return frozenset(l for l in DATA.links() if rng.random() >= r)


# ---------------------------------------------------------------- E0
HAND = {
    "decisions": ["가 취소 규정", "나 정렬", "다 사진"],
    "tasks": [("1", "가", "완료"), ("2", "가", "진행"), ("3", "가", "할 일"), ("4", "나", "완료"), ("5", "다", "진행")],
}


def e0_hand():
    out = {}
    for d in HAND["decisions"]:
        key = d.split()[0]
        linked = [t for t in HAND["tasks"] if t[1] == key]
        done = [t for t in linked if t[2] == "완료"]
        out[d] = {"linked": len(linked), "done": len(done), "open": len(linked) - len(done)}
    return out


# ---------------------------------------------------------------- E1
def e1_min_failure():
    tru = truth()
    pages = P.build_pages()
    ans, lines = P.answer_all(pages)
    db = R.build()
    rel = R.answer_all(db)
    return {
        "truth": tru,
        "pages": {q: {"answer": ans[q], "grade": grade(ans[q], tru[q]), "lines_read": lines[q]} for q in QS},
        "relational": {q: {"answer": rel[q], "grade": grade(rel[q], tru[q])} for q in QS},
        "relational_view_rows": R.view_rows(db),
        "pages_total_lines": P.total_lines(pages),
        "pages_lines_read_total": sum(lines.values()),
        "counts": {"decisions": len(DATA.DECISIONS), "tasks": len(DATA.TASKS), "specs": len(DATA.SPECS),
                   "links": len(DATA.links()),
                   "links_by_kind": {k: sum(1 for l in DATA.links() if l[0] == k) for k in ("owner", "spec", "task")}},
    }


# ---------------------------------------------------------------- E2, E3
def e2_rates():
    tru = truth()
    out = {}
    det_by_kind = {}
    for r in RATES:
        rng = random.Random(SEED)
        tally = {m: {q: {"correct": 0, "wrong": 0, "unanswerable": 0} for q in QS} for m in ("pages", "relational", "relational_fixed")}
        all5 = {m: 0 for m in tally}
        lost_n, det_n = 0, 0
        kind_lost = {"owner": 0, "spec": 0, "task": 0}
        kind_det = {"owner": 0, "spec": 0, "task": 0}
        for _ in range(N_SEEDS):
            lost = lose(rng, r)
            pa, _ = P.answer_all(P.build_pages(lost))
            db = R.build(lost)
            ra = R.answer_all(db)
            ra = as_graded(ra)
            found = R.detected_links(db, lost)
            fa = R.answer_all(R.build(lost - found))
            fa = as_graded(fa)
            for m, a in (("pages", pa), ("relational", ra), ("relational_fixed", fa)):
                gs = [grade(a[q], tru[q]) for q in QS]
                for q, g in zip(QS, gs):
                    tally[m][q][g] += 1
                all5[m] += gs.count("correct")
            lost_n += len(lost)
            det_n += len(found)
            for l in lost:
                kind_lost[l[0]] += 1
            for l in found:
                kind_det[l[0]] += 1
        out[str(r)] = {
            "tally": tally,
            "mean_correct_of_5": {m: all5[m] / N_SEEDS for m in all5},
            "mean_lost_links": lost_n / N_SEEDS,
            "detected_share": (det_n / lost_n) if lost_n else None,
        }
        det_by_kind[str(r)] = {k: {"lost": kind_lost[k], "detected": kind_det[k],
                                   "share": (kind_det[k] / kind_lost[k]) if kind_lost[k] else None} for k in kind_lost}
    return out, det_by_kind


def e3_example():
    """잃은 연결 하나씩: 빈칸 검사에 걸리는 것과 안 걸리는 것의 예."""
    cases = {
        "T10 근거 결정(D04) 하나뿐인 할 일": ("task", "D04", "T10"),
        "T13 근거 결정 둘 중 D10만 잃음": ("task", "D10", "T13"),
        "D04 영향 스펙 셋 중 S03만 잃음": ("spec", "D04", "S03"),
        "D01 영향 스펙 하나뿐(S03)": ("spec", "D01", "S03"),
        "D07 결정자": ("owner", "D07", "준호"),
    }
    out = {}
    tru = truth()
    for name, link in cases.items():
        lost = frozenset([link])
        db = R.build(lost)
        caught = bool(R.detected_links(db, lost))
        wrong = [q for q in QS if R.answer_all(db)[q] != tru[q]]
        out[name] = {"caught_by_empty_check": caught, "questions_changed": wrong}
    return out


# ---------------------------------------------------------------- E4
def e4_connection_trap():
    db = R.build()
    direct = R.q2(db)
    path = R.q2_via_tasks(db)
    return {
        "direct": direct, "via_tasks": path,
        "false_positive": sorted(set(path) - set(direct)), "missed": sorted(set(direct) - set(path)),
        "tasks_on_S03": sorted(t for t, s in db["task_spec"] if s == "S03"),
    }


# ---------------------------------------------------------------- E5
def e5_reversal():
    db = R.build()
    old, new = "D07", "D10"
    affected = sorted(t for t, d in db["task_dec"] if d == old)
    pages = P.build_pages()
    old_page_mentions = any("번복" in ln for ln in pages["decisions"][old])
    new_page_mentions = any(ln == f"번복: {old}" for ln in pages["decisions"][new])
    return {
        "old": old, "new": new,
        "affected_tasks": {t: db["tasks"][t]["status"] for t in affected},
        "affected_open": [t for t in affected if db["tasks"][t]["status"] != R.DONE],
        "affected_done_needs_rework": [t for t in affected if db["tasks"][t]["status"] == R.DONE],
        "pages_old_page_shows_reversal": old_page_mentions,
        "pages_new_page_shows_reversal": new_page_mentions,
        "two_way_relation_old_shows_reversal": True,
        "one_way_relation_old_shows_reversal": False,
    }


# ---------------------------------------------------------------- E6
def e6_drift(n_events=60, rates=(1.0, 0.95, 0.9, 0.8), seeds=N_SEEDS):
    """할 일 행·결정 행을 실제로 바꾸고, rollup은 매번 행에서 다시 센다. 손으로 적은 개수는 확률 p로만 고친다."""
    out = {}
    for p in rates:
        rng = random.Random(SEED + 60)
        mismatch_steps, final_mismatch, final_abs, rollup_mismatch = 0, 0, 0, 0
        for _ in range(seeds):
            db = R.build()
            start = R.project_rollups(db)
            m_dec, m_open = start["decisions"], start["open_tasks"]
            next_t, next_d = 21, 11
            for _ in range(n_events):
                ev = rng.choice(["done", "done", "new_task", "new_decision"])
                open_ids = sorted(t for t, v in db["tasks"].items() if v["status"] != R.DONE)
                if ev == "done" and open_ids:
                    db["tasks"][rng.choice(open_ids)]["status"] = R.DONE
                    if rng.random() < p:
                        m_open -= 1
                elif ev == "new_task":
                    t = f"T{next_t:02d}"
                    next_t += 1
                    db["tasks"][t] = {"title": "", "status": "할 일", "owner": "", "updated": "2026-10-01"}
                    db["task_dec"].add((t, rng.choice(sorted(db["decisions"]))))
                    if rng.random() < p:
                        m_open += 1
                elif ev == "new_decision":
                    d = f"D{next_d:02d}"
                    next_d += 1
                    db["decisions"][d] = {"title": "", "date": "2026-10-01", "owner": "", "supersedes": None}
                    if rng.random() < p:
                        m_dec += 1
                roll = R.project_rollups(db)
                count_rows = {"decisions": len(db["decisions"]),
                              "open_tasks": sum(1 for v in db["tasks"].values() if v["status"] != R.DONE)}
                rollup_mismatch += roll != count_rows
                if (m_dec, m_open) != (roll["decisions"], roll["open_tasks"]):
                    mismatch_steps += 1
            roll = R.project_rollups(db)
            if (m_dec, m_open) != (roll["decisions"], roll["open_tasks"]):
                final_mismatch += 1
            final_abs += abs(m_dec - roll["decisions"]) + abs(m_open - roll["open_tasks"])
        out[str(p)] = {
            "share_of_steps_mismatched": mismatch_steps / (seeds * n_events),
            "share_final_mismatched": final_mismatch / seeds,
            "mean_abs_error_final": final_abs / seeds,
            "rollup_mismatch_steps": rollup_mismatch,
            "n_events": n_events,
        }
    return out


# ---------------------------------------------------------------- E7
def random_db(rng):
    nd, nt, ns = rng.randint(3, 12), rng.randint(5, 30), rng.randint(2, 5)
    dates = [f"2026-09-{d:02d}" for d in range(1, 31)]
    db = {"decisions": {}, "tasks": {}, "specs": {}, "task_dec": set(), "dec_spec": set(), "task_spec": set()}
    for i in range(nd):
        db["decisions"][f"D{i:02d}"] = {"title": "", "date": rng.choice(dates), "owner": rng.choice(["a", "b", None]), "supersedes": None}
    for i in range(nt):
        db["tasks"][f"T{i:02d}"] = {"title": "", "status": rng.choice(["할 일", "진행", "검토", "완료", "완료"]),
                                    "owner": "a", "updated": rng.choice(dates)}
    for i in range(ns):
        db["specs"][f"S{i:02d}"] = {"title": "", "edited": rng.choice(dates)}
    D, T, S = list(db["decisions"]), list(db["tasks"]), list(db["specs"])
    for t in T:
        for d in rng.sample(D, rng.randint(0, min(2, len(D)))):
            db["task_dec"].add((t, d))
        for s in rng.sample(S, rng.randint(0, min(2, len(S)))):
            db["task_spec"].add((t, s))
    for d in D:
        for s in rng.sample(S, rng.randint(0, min(2, len(S)))):
            db["dec_spec"].add((d, s))
    return db, D[0], S[0]


def e7_cross(n=500):
    rng = random.Random(SEED + 7)
    mism = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0, "Q5": 0, "rollup_sql": 0, "rollup_formula": 0, "q2_via_tasks": 0}
    trap_differs = 0
    for _ in range(n):
        db, d1, s2 = random_db(rng)
        py = {"Q1": R.q1(db, d1), "Q2": R.q2(db, s2), "Q3": R.q3(db), "Q4": R.q4(db), "Q5": R.q5(db)}
        sq = R.sql_answer_all(R.to_sqlite(db), d1, s2)
        for q in py:
            mism[q] += py[q] != sq[q]
        roll = R.rollup_open_tasks(db)
        mism["rollup_sql"] += roll != sq["rollup_open"]
        mism["rollup_formula"] += roll != R.rollup_open_tasks_formula(db)
        mism["q2_via_tasks"] += R.q2_via_tasks(db, s2) != sq["q2_via_tasks"]
        trap_differs += R.q2(db, s2) != R.q2_via_tasks(db, s2)
    return {"n": n, "mismatches": mism, "trap_path_differs_from_direct": trap_differs}


def main():
    e2, e3k = e2_rates()
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "seed": SEED, "n_seeds": N_SEEDS, "rates": RATES, "checked": "2026-10-08"},
        "e0_hand": e0_hand(),
        "e1_min_failure": e1_min_failure(),
        "e2_rates": e2,
        "e3_detection_by_kind": e3k,
        "e3_examples": e3_example(),
        "e4_connection_trap": e4_connection_trap(),
        "e5_reversal": e5_reversal(),
        "e6_drift": e6_drift(),
        "e7_cross": e7_cross(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    data_dir = HERE / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "project.json").write_text(json.dumps(
        {"project": DATA.PROJECT, "decisions": DATA.DECISIONS, "tasks": DATA.TASKS, "specs": DATA.SPECS},
        ensure_ascii=False, indent=1))
    (data_dir / "pages_r1.md").write_text(_pages_md(P.build_pages()))
    print(json.dumps({k: res[k] for k in ("e1_min_failure",)}, ensure_ascii=False)[:1500])
    for r in RATES:
        print(r, res["e2_rates"][str(r)]["mean_correct_of_5"], res["e2_rates"][str(r)]["detected_share"])
    print(res["e4_connection_trap"])
    print(res["e6_drift"])
    print(res["e7_cross"])


def _pages_md(pages):
    out = []
    for d, lines in pages["decisions"].items():
        out += lines + [""]
    out += pages["task_list"] + [f"(마지막 수정 {pages['task_list_edited']})", ""]
    for s, v in pages["specs"].items():
        out += v["lines"] + [f"(마지막 수정 {v['edited']})", ""]
    return "\n".join(out)


if __name__ == "__main__":
    main()
