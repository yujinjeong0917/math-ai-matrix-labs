"""내 데이터로 LLM 파인튜닝하기 1장 실험. `uv run python finetune/ch01_when_finetune/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 문의 10건(K3 F2 T2 I2 R1)에 방법 묶음을 대면 몇 건이 남나
E1 토큰 예산: 예시 k개를 넣은 호출 길이, 문맥 창별 k_max, 한 달 토큰
E2 검색기: K 문의 12건이 정답 문서를 1등으로 가져오나(정책 v1, v2)
E3 문의 40건 비교(문맥 창 8,192, 정책 v1): 방법 묶음 8가지의 남는 실패
E4 정책 변경 뒤(v2, 파인튜닝은 v1로 학습한 채): 옛 정책으로 틀리는 답
E5 문맥 창 131,072일 때: 프롬프트가 말투까지 고치지만 호출이 길어진다
E6 가정 민감도: 파인튜닝이 사실을 배우는 기준 횟수(min_fact_count)를 1~6으로 바꾸면
E7 규칙 표의 근거 세기: 칸마다 URL이 있는지, [가정]인지

표준 라이브러리만 쓴다. 무작위가 없어 결과는 매번 같다.
"""

import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import finetune_ch01_compare as C  # noqa: E402
import finetune_ch01_retrieve as RT  # noqa: E402

OUT = HERE / "results" / "ch01.json"
NAMES = {"": "아무것도 안 함", "P": "프롬프트", "R": "검색(RAG)", "F": "파인튜닝",
         "PR": "프롬프트 + 검색", "PF": "프롬프트 + 파인튜닝", "RF": "검색 + 파인튜닝", "PRF": "셋 다"}


def e0_hand():
    c = {"K": 3, "F": 2, "T": 2, "I": 2, "R": 1, "OK": 0}
    # 손 예시의 가정: 검색은 K 3건을 모두 찾고, 파인튜닝은 K 3건 중 1건만 지난 답변에 충분히 나왔다.
    fixes = {"P": {"F": 2, "I": 2}, "R": {"K": 3}, "F": {"F": 2, "T": 2, "I": 2, "K": 1}}
    rows = []
    for combo in ["P", "R", "F", "PR", "RF"]:
        merged = {}
        for m in combo:
            for t, n in fixes[m].items():
                merged[t] = max(merged.get(t, 0), n)
        rows.append({"combo": combo, "fixed": sum(merged.values()), "remaining": C.hand_remaining(c, merged)})
    return {"counts": c, "total": sum(c.values()), "fixes": fixes, "rows": rows}


def e1_budget(b):
    s, e, q, a = b["system_tokens"], b["example_tokens"], b["question_tokens"], b["answer_reserve_tokens"]
    k = b["hand"]["k"]
    hand = C.tokens_per_call(s, k, e, q)
    win = []
    for w in b["windows"]:
        km = C.k_max(w, s, e, q, a)
        win.append({"window": w, "room": w - s - q - a, "k_max": km,
                    "fits_needed_T": km >= C.load_rules()["params"]["needed_examples_T"]})
    nT = C.load_rules()["params"]["needed_examples_T"]
    per_call_T = C.tokens_per_call(s, nT, e, q)
    per_call_ft = b["finetuned_system_tokens"] + q
    mc = b["monthly_calls"]
    return {"s": s, "e": e, "q": q, "a": a, "hand_k": k, "hand_tokens": hand, "windows": win,
            "needed_examples_T": nT, "per_call_with_needed_T": per_call_T, "per_call_finetuned": per_call_ft,
            "monthly_calls": mc, "monthly_tokens_prompt_T": per_call_T * mc, "monthly_tokens_finetuned": per_call_ft * mc,
            "ratio_prompt_over_ft": round(per_call_T / per_call_ft, 1)}


def e2_retrieval(items, v1, v2):
    rows = []
    for it in items:
        if it["failure"] != "K":
            continue
        r1 = RT.rank(it["text"], v1["docs"])
        top = RT.top1(it["text"], v1["docs"])
        gold_doc = next(d for d in v1["docs"] if d["id"] == it["gold"])
        gs = RT.score(it["text"], gold_doc)
        reason = "hit" if top == it["gold"] else ("gold_no_overlap" if gs == 0 else ("tie" if top is None else "wrong_doc"))
        rows.append({"id": it["id"], "text": it["text"], "gold": it["gold"], "top": top,
                     "top3": [[s, d] for s, d in r1[:3]], "gold_score": RT.score(it["text"], gold_doc),
                     "shared_with_gold": RT.shared(it["text"], gold_doc), "hit": top == it["gold"],
                     "hit_v2": RT.top1(it["text"], v2["docs"]) == it["gold"], "reason": reason})
    hits = sum(r["hit"] for r in rows)
    return {"rows": rows, "k_items": len(rows), "hits": hits, "hits_v2": sum(r["hit_v2"] for r in rows),
            "hit_rate": round(hits / len(rows), 4)}


def table(model, items):
    out = []
    for combo in C.COMBOS:
        t = model.tally(items, combo)
        out.append({"combo": combo, "name": NAMES[combo], "ok": t["ok"], "fail": t["fail"], "stale": t["stale"],
                    "remaining": t["remaining"],
                    "remaining_by_type": {k: v["fail"] + v["stale"] for k, v in t["by_type"].items() if v["fail"] + v["stale"]}})
    return out


def e3_compare(items, v1):
    m = C.Model(v1, window=8192)
    ft_known = sorted({it["gold"] for it in items if it["failure"] == "K" and m.ft_knows(it)})
    return {"window": 8192, "k_max": m.kmax(), "counts": C.counts(items), "total": len(items),
            "ft_known_docs": ft_known, "rows": table(m, items)}


def e4_policy_change(items, v1, v2):
    m = C.Model(v2, trained_on=v1, window=8192)
    changed = [d["id"] for d, o in zip(v2["docs"], v1["docs"]) if d["text"] != o["text"]]
    stale_items = [it["id"] for it in items if m.outcome(it, "F") == "stale"]
    stale_rf = [it["id"] for it in items if m.outcome(it, "RF") == "stale"]
    return {"changed_docs": changed, "stale_items_F": stale_items, "stale_items_RF": stale_rf, "rows": table(m, items)}


def e5_big_window(items, v1, b):
    m = C.Model(v1, window=131072)
    return {"window": 131072, "k_max": m.kmax(), "rows": table(m, items)}


def e6_sensitivity(items, v1):
    out = []
    base = C.load_rules()
    for n in range(1, 7):
        rules = json.loads(json.dumps(base))
        rules["params"]["min_fact_count"] = n
        m = C.Model(v1, rules=rules)
        m2 = C.Model(C.load("policies_v2.json"), rules=rules, trained_on=v1)
        out.append({"min_fact_count": n, "F_remaining_v1": m.tally(items, "F")["remaining"],
                    "F_stale_v2": m2.tally(items, "F")["stale"], "RF_remaining_v2": m2.tally(items, "RF")["remaining"]})
    return out


def e7_basis():
    rules = C.load_rules()["rules"]
    cells = []
    for m in "PRF":
        for t in ["K", "F", "T", "I", "R"]:
            c = rules[m][t]
            cells.append({"method": m, "type": t, "fix": c["fix"], "has_url": "url" in c,
                          "assumed": c["basis"].startswith("[가정]")})
    return {"cells": len(cells), "with_url": sum(c["has_url"] for c in cells),
            "assumed_only": sum(c["assumed"] and not c["has_url"] for c in cells),
            "computed": sum(c["fix"] in ("budget", "retrieval", "count") for c in cells), "detail": cells}


def main():
    v1, v2 = C.load("policies_v1.json"), C.load("policies_v2.json")
    items = C.load("inquiries.json")["items"]
    b = C.load("budget.json")
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "rule_file": C.RULE_FILE.name, "checked": "2026-10-11"},
        "e0_hand": e0_hand(),
        "e1_budget": e1_budget(b),
        "e2_retrieval": e2_retrieval(items, v1, v2),
        "e3_compare": e3_compare(items, v1),
        "e4_policy_change": e4_policy_change(items, v1, v2),
        "e5_big_window": e5_big_window(items, v1, b),
        "e6_sensitivity": e6_sensitivity(items, v1),
        "e7_basis": e7_basis(),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    print(json.dumps({k: v for k, v in r.items() if k not in ("e2_retrieval",)}, ensure_ascii=False, indent=1)[:6000])
    for row in r["e2_retrieval"]["rows"]:
        print(row["id"], row["gold"], row["top"], row["reason"], row["top3"], row["shared_with_gold"])
