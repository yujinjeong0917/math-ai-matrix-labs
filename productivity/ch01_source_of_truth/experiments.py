"""생산성 도구 1장 비교 실험. `OMP_NUM_THREADS=2 uv run python productivity/ch01_source_of_truth/experiments.py` 로 실행한다.

E0 첫 화면 손계산: q=0.9, 자리 3곳, 변경 8번
E1 실패 최소 예제: 값 3개 x 자리 3곳, 카드 8장을 "한 곳만 고친다"로 복사 방식에 넣기
E2 어느 쪽이 맞나: 다수결, 파일 수정 시각, 뒤늦게 정한 규칙 자리로 골라 보기
E3 같은 카드 8장을 원본 등록부 방식에 넣기(손 옮김 안 함 / 함)
E4 변경 1건당 고칠 자리 수
E5 무작위 카드 8장 x 시드 5000개, q 네 단계: 복사 / 복사(함께 놓침) / 등록부 / 등록부(모두 링크)
E6 일반 모형 k=1~5: 식, 순수 파이썬, NumPy 세 구현 대조
E7 원본 등록부 15항목 점검표, 일부러 망가뜨린 등록부

고정 시드라 결과는 매번 같다. 스레드는 2개 이하.
"""

import copy
import csv
import json
import os
import platform
import random
import statistics
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import productivity_ch01_np as N  # noqa: E402
import productivity_ch01_project as P  # noqa: E402
import productivity_ch01_sim as S  # noqa: E402

OUT = HERE / "results" / "ch01.json"
CHECKED = "2026-10-10"
QS = [0.7, 0.8, 0.9, 0.95]
SEEDS = 5000


def table(state):
    return {P.VALUE_LABEL[v]: {P.PLACE_LABEL[p]: P.render(v, state[p][v]) for p in P.PLACES} for v in P.VALUES}


def e0_hand():
    q, k, m = P.HAND["q"], P.HAND["k"], P.HAND["m"]
    return {
        "q": q, "k": k, "m": m,
        "one_change_all_right": round(q ** (k - 1), 4),          # 0.81
        "after_m_never_wrong": round((q ** (k - 1)) ** m, 4),     # 0.81^8
        "k2_after_m": round(q ** m, 4),                           # 손으로 옮기는 곳이 하나면
        "k1_after_m": 1.0,
    }


def e1_copy():
    r = S.run_copy(P.CARDS)
    _, hist = S._truth_history(P.CARDS)
    cells = []
    for v in P.VALUES:
        for p in P.PLACES:
            cells.append({"value": P.VALUE_LABEL[v], "place": P.PLACE_LABEL[p], "shown": P.render(v, r["state"][p][v]),
                          "right": r["state"][p][v] == r["truth"][v], "ever_true": r["state"][p][v] in hist[v]})
    return {
        "initial": {P.VALUE_LABEL[v]: P.render(v, P.INITIAL[v]) for v in P.VALUES},
        "truth": {P.VALUE_LABEL[v]: P.render(v, r["truth"][v]) for v in P.VALUES},
        "truth_history": {P.VALUE_LABEL[v]: [P.render(v, x) for x in hist[v]] for v in P.VALUES},
        "final_table": table(r["state"]),
        "cells": cells,
        "mismatch_cells": r["per_step"][-1],
        "per_step": r["per_step"],
        "distinct_values": {P.VALUE_LABEL[v]: n for v, n in S.distinct_values(r["state"]).items()},
        "truth_anywhere": {P.VALUE_LABEL[v]: a for v, a in S.truth_anywhere(r["state"], r["truth"]).items()},
        "never_true_cells": S.never_true_cells(r["state"], P.CARDS),
        "edits_made": len(P.CARDS),
        "last_edit_card": {P.PLACE_LABEL[p]: P.CARDS[i]["id"] if i >= 0 else None for p, i in r["last_edit"].items()},
    }


def e2_judge():
    r = S.run_copy(P.CARDS)
    return S.judge(r)


def e3_register():
    none = S.run_register(P.CARDS)
    done = S.run_register(P.CARDS, lambda c, p: True)
    full_copy = S.run_copy(P.CARDS, lambda c, p: True)
    return {
        "one_place_rule": {
            "final_table": table(none["state"]), "mismatch_cells": none["per_step"][-1], "per_step": none["per_step"],
            "source_right": sum(none["state"][P.SOURCE[v]][v] == none["truth"][v] for v in P.VALUES),
            "never_true_cells": S.never_true_cells(none["state"], P.CARDS),
            "stale_hand_cells": [f"{P.VALUE_LABEL[v]}@{P.PLACE_LABEL[p]}" for v in P.VALUES for p in P.PLACES
                                 if none["state"][p][v] != none["truth"][v]],
        },
        "with_hand_copies": {"mismatch_cells": done["per_step"][-1], "per_step": done["per_step"]},
        "copy_everyone_fixes_all": {"mismatch_cells": full_copy["per_step"][-1]},
        "hand_links": {P.VALUE_LABEL[v]: sum(h == "hand" for h in P.LINKS[v].values()) for v in P.VALUES},
        "ref_links": {P.VALUE_LABEL[v]: sum(h == "ref" for h in P.LINKS[v].values()) for v in P.VALUES},
    }


def e4_edits():
    per_value = {}
    for v in P.VALUES:
        c = {"value": v}
        per_value[P.VALUE_LABEL[v]] = {"copy": S.edits_per_card(c, "copy"), "register": S.edits_per_card(c, "register")}
    return {
        "per_value": per_value,
        "cards_per_value": {P.VALUE_LABEL[v]: sum(c["value"] == v for c in P.CARDS) for v in P.VALUES},
        "total_8_cards": {"copy": sum(S.edits_per_card(c, "copy") for c in P.CARDS),
                          "register": sum(S.edits_per_card(c, "register") for c in P.CARDS),
                          "register_ideal": len(P.CARDS)},
        "places_to_remember_per_change": {"copy": "세 자리 모두(머릿속)", "register": "등록부에 적힌 자리"},
    }


def e5_montecarlo():
    out = {}
    modes = [("copy", False), ("copy_correlated", True), ("register", False), ("register_ideal", False)]
    for q in QS:
        row = {}
        for name, corr in modes:
            mode = "copy" if name.startswith("copy") else name
            rs = [S.simulate(mode, q, random.Random(10_000 * int(q * 100) + s), correlated=corr) for s in range(SEEDS)]
            row[name] = {
                "final_mismatch": round(statistics.mean(r["final_mismatch"] for r in rs), 3),
                "p_any_final": round(statistics.mean(not r["final_clean"] for r in rs), 4),
                "p_ever_clean": round(statistics.mean(r["ever_clean"] for r in rs), 4),
                "exposure": round(statistics.mean(r["exposure"] for r in rs), 3),
                "never_true": round(statistics.mean(r["never_true"] for r in rs), 3),
                "values_lost": round(statistics.mean(r["values_lost"] for r in rs), 3),
                "majority_right": round(statistics.mean(r["majority_right"] for r in rs), 3),
                "source_right": round(statistics.mean(r["source_right"] for r in rs), 3),
            }
        out[str(q)] = row
    return out


def e6_generic():
    rows = []
    m = 8
    for corr in (False, True):
        for k in range(1, 6):
            for q in QS:
                ex = S.generic_exact(k, q, m, corr)
                py = S.generic_python(k, q, m, 20_000, seed=k * 1000 + int(q * 100) + (7 if corr else 0), correlated=corr)
                npv = N.generic_numpy(k, q, m, 200_000, seed=k * 1000 + int(q * 100) + (7 if corr else 0), correlated=corr)
                rows.append({"correlated": corr, "k": k, "q": q, "m": m,
                             "exact": [round(x, 4) for x in ex], "python": [round(x, 4) for x in py],
                             "numpy": [round(x, 4) for x in npv]})
    worst_py = max(abs(a - b) for r in rows for a, b in zip(r["exact"], r["python"]))
    worst_np = max(abs(a - b) for r in rows for a, b in zip(r["exact"], r["numpy"]))
    return {"columns": ["P(한 번도 안 어긋남)", "P(끝에 다 맞음)", "변경당 어긋난 자리"],
            "rows": rows, "max_abs_gap_python": round(worst_py, 4), "max_abs_gap_numpy": round(worst_np, 4),
            "n_python": 20_000, "n_numpy": 200_000}


def broken_register():
    """흔히 보는 모양으로 망가뜨린 등록부: 요금 원본이 둘, 운영 시간을 운영 시트에 다시 적음, 화면 구성을 Notion에 둠, 알림 없음."""
    reg = copy.deepcopy(P.REGISTER)
    by = {it["id"]: it for it in reg}
    reg.append({"id": "I06b", "name": "시간당 요금", "kind": "number", "tool": "sheets", "where": "운영표!요금", "owner": "하린",
                "refs": [], "notify": "시트 변경 알림"})
    by["I05"]["refs"].append(("sheets", "운영표!영업시간", "typed"))
    by["I14"]["tool"] = "notion"
    by["I02"]["notify"] = ""
    by["I11"]["owner"] = "누군가"
    return reg


def e7_lint():
    good = S.lint_register(P.REGISTER)
    bad = S.lint_register(broken_register())
    return {"items": len(P.REGISTER), "refs": S.count_refs(P.REGISTER),
            "max_rows_per_name": max(sum(x["name"] == it["name"] for x in P.REGISTER) for it in P.REGISTER),
            "kinds": {k: sum(it["kind"] == k for it in P.REGISTER) for k in P.DECISION_RULE},
            "good_problems": good, "broken_problems": [list(x) for x in bad], "broken_count": len(bad)}


def write_data():
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "data" / "cards.json").write_text(json.dumps(
        {"places": P.PLACE_LABEL, "values": P.VALUE_LABEL, "initial": P.INITIAL, "cards": P.CARDS,
         "source": P.SOURCE, "links": P.LINKS}, ensure_ascii=False, indent=1))
    (HERE / "data" / "register.json").write_text(json.dumps(P.REGISTER, ensure_ascii=False, indent=1))


def write_checks(e1, e3, e4):
    (HERE / "checks").mkdir(exist_ok=True)
    with open(HERE / "checks" / "ch01.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["reader", "date", "method", "mismatch_cells_of_9", "values_with_truth_nowhere", "never_true_cells",
                    "edits_to_fully_sync_8_cards", "note"])
        lost = sum(not a for a in e1["truth_anywhere"].values())
        w.writerow(["example", CHECKED, "copy", e1["mismatch_cells"], lost, e1["never_true_cells"], e4["total_8_cards"]["copy"],
                    "합성 모형 결과. 카드 8장을 받은 자리 한 곳만 고침"])
        w.writerow(["example", CHECKED, "register", e3["one_place_rule"]["mismatch_cells"], 0,
                    e3["one_place_rule"]["never_true_cells"], e4["total_8_cards"]["register"],
                    "합성 모형 결과. 원본 한 곳만 고침(손 옮김 안 함). 독자는 자기 기록을 한 줄씩 더한다"])


def main():
    write_data()
    e1, e3, e4 = e1_copy(), e3_register(), e4_edits()
    write_checks(e1, e3, e4)
    res = {
        "meta": {"python": platform.python_version(), "numpy": np.__version__,
                 "platform": f"{platform.system()} {platform.machine()}", "checked": CHECKED,
                 "seeds": SEEDS, "qs": QS, "cards": len(P.CARDS)},
        "e0_hand": e0_hand(),
        "e1_copy": e1,
        "e2_judge": e2_judge(),
        "e3_register": e3,
        "e4_edits": e4,
        "e5_montecarlo": e5_montecarlo(),
        "e6_generic": e6_generic(),
        "e7_lint": e7_lint(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "e6_generic"}, ensure_ascii=False, indent=1))
    print("e6 gaps", res["e6_generic"]["max_abs_gap_python"], res["e6_generic"]["max_abs_gap_numpy"])


if __name__ == "__main__":
    main()
