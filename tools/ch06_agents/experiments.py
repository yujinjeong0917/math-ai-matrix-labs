"""AI 도구 실전 6장 실험. `uv run python tools/ch06_agents/experiments.py` 로 실행한다.

E0 첫 화면 손계산: p = 0.97로 2, 3, 22단계, 90%를 지키는 최대 단계 수, 한 번 다시 하기
E1 관통 프로젝트의 주간 반복: data/weekly_run.json의 단계 수와 종류
E2 이어 붙인 성공률 표 p^n
E3 확인 지점(검증 테스트) + 다시 하기 + 사람 승인: 닫힌 식
E4 승인 횟수: 모든 단계 승인 / 되돌리기 어려운 단계만 / 승인 없음
E5 워크플로와 에이전트: 보통 주와 이상한 주(문의 3쪽), 같아지는 비율
E6 시뮬레이션(고정 시드 6): 닫힌 식과 표준오차 3배 안에서 맞는지
E7 멈추는 자리: 처음 틀리는 단계 분포, 다시 하기가 늘리는 Figma 호출 수
E8 대조: 무작위 300경우에서 닫힌 식과 하나씩 펼쳐 센 값
E9 설정 예시 읽기: Claude Code settings.json, Codex config.toml

표준 라이브러리만 쓴다. 네트워크·API 키가 필요 없고 결과는 매번 같다.
"""

import json
import math
import platform
import random
import sys
import tomllib
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import tools_ch06_chain as C  # noqa: E402
import tools_ch06_loop as L  # noqa: E402

OUT = HERE / "results" / "ch06.json"
FACTS = HERE / "data" / "agents_2026-10-10.json"
WEEKS = 20_000


def f(x, nd=6):
    return round(float(x), nd)


def outs(o):
    return {k: f(v) for k, v in o.items() if k != "stop_at"}


def e0_hand(a):
    p = a["p"]
    return {"p": f(p), "n2": f(C.chain(p, 2)), "n3": f(C.chain(p, 3)), "n4": f(C.chain(p, 4)),
            "n22": f(C.chain(p, 22), 4), "max_steps_90": C.max_steps(p, F(9, 10)),
            "retry_once_step": f(C.per_step_retry_perfect(p, 1)),
            "retry_once_n22": f(C.per_step_retry_perfect(p, 1) ** 22, 4),
            "fail_twice": f((1 - p) ** 2)}


def e1_project(run):
    st = run["steps"]
    kinds = {}
    for s in st:
        kinds[s["kind"]] = kinds.get(s["kind"], 0) + 1
    return {"n": len(st), "kinds": kinds, "hard_undo": [s["id"] for s in st if s["undo"] == "hard"],
            "tools": sorted({s["tool"] for s in st if s["tool"]}),
            "notion_reads": sum(s["id"].startswith("notion_") for s in st),
            "figma_reads": sum(s["id"].startswith("figma_") for s in st),
            "sections": sum(s["id"].startswith("section_") for s in st),
            "pages_normal": math.ceil(run["rows_normal"] / run["rows_per_page"]),
            "pages_odd": math.ceil(run["rows_odd"] / run["rows_per_page"])}


def e2_table():
    ps = [F(90, 100), F(95, 100), F(97, 100), F(99, 100), F(999, 1000)]
    ns = [1, 5, 10, 22, 50]
    return {"ps": [f(p) for p in ps], "ns": ns,
            "rows": [[f(C.chain(p, n), 4) for n in ns] for p in ps],
            "max_steps_90": [C.max_steps(p, F(9, 10)) for p in ps]}


def scen(a, n=22):
    p, d, k, h = a["p"], a["d"], a["k"], a["h"]
    gate = n - 1
    return {
        "none": C.run_outcomes(C.uniform(n, p)),
        "check_no_retry": C.run_outcomes(C.uniform(n, p, d, 0)),
        "check_retry": C.run_outcomes(C.uniform(n, p, d, k)),
        "check_retry_human": C.run_outcomes(C.uniform(n, p, d, k), human=h, gate=gate),
        "perfect_check_retry1": C.run_outcomes(C.uniform(n, p, 1, 1)),
    }


def e3_checks(a):
    so = C.step_outcomes(a["p"], a["d"], a["k"])
    return {"assume": {k: (f(v) if isinstance(v, F) else v) for k, v in a.items()},
            "step": {k: f(v, 8) for k, v in so.items()},
            "caught_per_try": f((1 - a["p"]) * a["d"]),
            "missed_per_try": f((1 - a["p"]) * (1 - a["d"])),
            "scenarios": {k: outs(v) for k, v in scen(a).items()}}


def e4_approvals(run):
    n = len(run["steps"])
    hard = sum(s["undo"] == "hard" for s in run["steps"])
    writes = sum(s["kind"] == "write" for s in run["steps"])
    return {"every_step_week": n, "every_step_year": 52 * n, "writes_week": writes, "writes_year": 52 * writes,
            "hard_only_week": hard, "hard_only_year": 52 * hard}


def e5_agent(a, n=22):
    r = C.agent_vs_workflow(a["p"], a["q"], n, a["f"])
    pq = a["p"] * a["q"]
    return {**{k: f(v, 4) for k, v in r.items()}, "pq": f(pq, 4),
            "agent_check_human_normal": outs(C.run_outcomes(C.uniform(n, pq, a["d"], a["k"]), human=a["h"], gate=n - 1)),
            "agent_check_human_odd": outs(C.run_outcomes(C.uniform(n + 1, pq, a["d"], a["k"]), human=a["h"], gate=n)),
            "workflow_check_odd_stop_at_notion2": f(sum(C.step_outcomes(a["p"], a["d"], a["k"])[x]
                                                        for x in ("correct", "missed")) ** 2, 4)}


def sim_rows(a):
    p, q, d, k, h = (float(a[x]) for x in ("p", "q", "d", "k", "h"))
    k = int(k)
    n = 22
    pq = a["p"] * a["q"]
    plan = [
        ("workflow_none", L.run_workflow, None, dict(p=p), C.run_outcomes(C.uniform(n, a["p"]))),
        ("workflow_check_retry", L.run_workflow, None, dict(p=p, d=d, k=k, checks=True),
         C.run_outcomes(C.uniform(n, a["p"], a["d"], k))),
        ("workflow_check_retry_human", L.run_workflow, None, dict(p=p, d=d, k=k, checks=True, human=True, h=h),
         C.run_outcomes(C.uniform(n, a["p"], a["d"], k), human=a["h"], gate=n - 1)),
        ("agent_none", L.run_agent, None, dict(p=p, q=q), C.run_outcomes(C.uniform(n, pq))),
        ("agent_none_odd", L.run_agent, 1, dict(p=p, q=q), C.run_outcomes(C.uniform(n + 1, pq))),
        ("agent_check_retry_human", L.run_agent, None, dict(p=p, q=q, d=d, k=k, checks=True, human=True, h=h),
         C.run_outcomes(C.uniform(n, pq, a["d"], k), human=a["h"], gate=n - 1)),
        ("workflow_none_odd", L.run_workflow, 1, dict(p=p), None),
        ("workflow_check_odd", L.run_workflow, 1, dict(p=p, d=d, k=k, checks=True), None),
    ]
    rows = []
    for name, fn, odd, kw, cf in plan:
        s = L.simulate(fn, WEEKS, odd_share=odd, **kw)
        row = {"name": name, **{x: s[x] for x in ("weeks", "correct", "wrong", "stopped", "attempts", "approvals")},
               "correct_rate": f(s["correct"] / WEEKS, 4), "wrong_rate": f(s["wrong"] / WEEKS, 4),
               "stopped_rate": f(s["stopped"] / WEEKS, 4), "stop_steps": dict(sorted(s["stop_steps"].items()))}
        if cf is not None:
            pc = float(cf["all_correct"])
            se = math.sqrt(pc * (1 - pc) / WEEKS)
            row.update(closed_correct=f(pc, 4), se=f(se, 5), z=f((s["correct"] / WEEKS - pc) / se, 2) if se else 0.0)
        rows.append(row)
    return rows


def e7_where(a, run):
    p = a["p"]
    ids = [s["id"] for s in run["steps"]]
    first = [f(C.first_failure_at(p, i), 4) for i in range(1, 23)]
    so = C.step_outcomes(a["p"], a["d"], a["k"])
    figma_runs_month = F(52, 12)
    return {"first_failure_at": dict(zip(ids, first)), "first_failure_at_1": f(C.first_failure_at(p, 1), 4),
            "first_failure_at_22": f(C.first_failure_at(p, 22), 4),
            "first_failure_in_sections": f(sum(C.first_failure_at(p, i) for i in range(9, 21)), 4),
            "no_failure": f(C.chain(p, 22), 4),
            "attempts_per_step": f(so["attempts"], 6),
            "figma_calls_month_no_retry": f(4 * figma_runs_month, 2),
            "figma_calls_month_retry": f(4 * so["attempts"] * figma_runs_month, 2),
            "figma_limit_month": 20}


def enumerate_paths(p, d, k, n):
    """시도 하나하나를 나무처럼 펼쳐 모든 경로의 확률을 더한다(작은 n만)."""
    tries = [("ok", p), ("caught", (1 - p) * d), ("missed", (1 - p) * (1 - d))]
    tot = {"correct": F(0), "wrong": F(0), "stopped": F(0)}

    def walk(i, used, wrong, prob):
        if prob == 0:
            return
        if i == n:
            tot["wrong" if wrong else "correct"] += prob
            return
        for kind, pr in tries:
            if kind == "ok":
                walk(i + 1, 0, wrong, prob * pr)
            elif kind == "missed":
                walk(i + 1, 0, True, prob * pr)
            elif used < k:
                walk(i, used + 1, wrong, prob * pr)
            else:
                tot["stopped"] += prob * pr

    walk(0, 0, False, F(1))
    return tot


def e8_crosscheck(n_cases=300):
    rng = random.Random(6)
    bad = 0
    for _ in range(n_cases):
        p = F(rng.randint(80, 100), 100)
        d = F(rng.randint(0, 10), 10)
        k = rng.randint(0, 3)
        n = rng.randint(1, 5)
        o = C.run_outcomes(C.uniform(n, p, d, k))
        e = enumerate_paths(p, d, k, n)
        if not (o["all_correct"] == e["correct"] and o["delivered_wrong"] == e["wrong"]
                and o["stopped_by_check"] == e["stopped"] and sum(e.values()) == 1):
            bad += 1
        if d == 0 and k == 0 and o["all_correct"] != p ** n:
            bad += 1
    return {"cases": n_cases, "mismatches": bad}


def e9_configs():
    cs = json.loads((HERE / "data" / "claude_settings.example.json").read_text(encoding="utf-8"))
    cx = tomllib.loads((HERE / "data" / "codex_config.example.toml").read_text(encoding="utf-8"))
    return {"claude_settings": cs, "codex_config": cx}


def main():
    run = C.load_run()
    a = C.assumptions(run)
    facts = json.loads(FACTS.read_text(encoding="utf-8"))
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "seed": L.SEED, "weeks": WEEKS, "facts_file": FACTS.name, "checked": facts["checked"]},
        "e0_hand": e0_hand(a),
        "e1_project": e1_project(run),
        "e2_table": e2_table(),
        "e3_checks": e3_checks(a),
        "e4_approvals": e4_approvals(run),
        "e5_agent": e5_agent(a),
        "e6_sim": sim_rows(a),
        "e7_where": e7_where(a, run),
        "e8_crosscheck": e8_crosscheck(),
        "e9_configs": e9_configs(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
