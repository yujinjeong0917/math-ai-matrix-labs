"""내 데이터로 LLM 파인튜닝하기 9장 실험. `uv run python finetune/ch09_evaluate/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 검증 손실 6개로 조기 종료(인내 2), 엇갈린 문항 2개의 동전 던지기
E1 가상 손실 곡선(6에폭, 17스텝마다 평가): 가장 낮은 지점, 멈추는 지점, 8장 설정(34스텝마다·3에폭)이면
E2 조기 종료 threshold가 바꾸는 것(손계산 표에 0.02, 0.05)
E3 생성 설정: 무작위로 뽑아 쓰면 같은 모델의 통과 수가 얼마나 흔들리나(장난감)
E4 채점 방법별 통과 수: 정확 일치, 정규화 일치, 규칙 검사, 사람 채점표(base, ft_best, ft_last)
E5 실패 종류별 바뀜(base -> ft_best), 1장 모형의 예상과 나란히
E6 일반 문항 20개로 본 잊어버림
E7 장난감 채점자: 보여 주는 순서를 바꾸면
E8 우연인가: 짝지은 부트스트랩 1,000번과 부호 검정, 몇 문항이면 갈리나

표준 라이브러리만 쓴다. 무작위는 고정 시드라 결과가 매번 같다. 모든 채점 데이터는 가상이다.
"""

import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ft_ch09_eval as M  # noqa: E402

OUT = HERE / "results" / "ch09.json"
CH01 = HERE.parent / "ch01_when_finetune" / "results" / "ch01.json"


def r4(x):
    return round(x, 4)


def e0_hand(sc):
    tr = M.early_stopping_trace(sc["hand_curve"], patience=sc["early_stopping"]["patience"])
    return {"values": sc["hand_curve"], "trace": tr["rows"], "stop_eval": tr["stop"] + 1, "best_eval": tr["best"] + 1,
            "coin_two_same_side_one_sided": 0.25, "coin_two_two_sided": M.sign_test(0, 2)}


def e1_curves(sc):
    cfg, spe = sc["curve"], sc["from_ch08"]["steps_per_epoch"]
    cv = M.make_curves(cfg, spe)
    ev = cv["evals"]
    vals = [e["val"] for e in ev]
    p = sc["early_stopping"]["patience"]
    tr = M.early_stopping_trace(vals, patience=p)
    best = ev[tr["best"]]
    stop = ev[tr["stop"]] if tr["stop"] is not None else None
    # 8장 설정처럼 34스텝마다만 평가하면(6에폭까지 학습한다고 둘 때와, 노트북대로 3에폭에서 끝날 때)
    every_epoch = [e for e in ev if e["step"] % spe == 0]
    tr_ep = M.early_stopping_trace([e["val"] for e in every_epoch], patience=p)
    ch08 = [e for e in every_epoch if e["step"] <= sc["from_ch08"]["epochs_in_ch08_notebook"] * spe]
    tr_08 = M.early_stopping_trace([e["val"] for e in ch08], patience=p)
    leaky = [e["leaky_val"] for e in ev]
    tr_leak = M.early_stopping_trace(leaky, patience=p)
    smooth_train = [M.train_loss(s, cfg["train"]) for s in range(1, cv["total_steps"] + 1)]
    below = M.first_below(smooth_train, sc["unsloth_rule_train_loss"])
    noisy_below = M.first_below(cv["train"], sc["unsloth_rule_train_loss"])
    last = ev[-1]
    return {
        "total_steps": cv["total_steps"], "eval_every": cfg["eval_every"], "evals": ev,
        "train_every_10": [[s, cv["train"][s - 1]] for s in range(10, cv["total_steps"] + 1, 10)],
        "best_step": best["step"], "best_val": best["val"], "best_epoch": r4(best["step"] / spe),
        "stop_step": stop["step"] if stop else None, "stop_val": stop["val"] if stop else None,
        "steps_saved": cv["total_steps"] - stop["step"] if stop else 0,
        "gap_at_best": r4(best["val"] - best["train_smooth"]), "gap_at_last": r4(last["val"] - last["train_smooth"]),
        "last_step": last["step"], "last_val": last["val"], "last_train_smooth": last["train_smooth"],
        "val_rise_best_to_last": r4(last["val"] - best["val"]),
        "trace": tr["rows"],
        "every_epoch": {"steps": [e["step"] for e in every_epoch], "vals": [e["val"] for e in every_epoch],
                        "best_step": every_epoch[tr_ep["best"]]["step"],
                        "stop_step": every_epoch[tr_ep["stop"]]["step"] if tr_ep["stop"] is not None else None},
        "ch08_notebook": {"steps": [e["step"] for e in ch08], "vals": [e["val"] for e in ch08],
                          "best_step": ch08[tr_08["best"]]["step"], "stopped": tr_08["stop"] is not None,
                          "last_step": ch08[-1]["step"]},
        "leaky": {"min_step": ev[min(range(len(leaky)), key=lambda i: leaky[i])]["step"],
                  "stopped": tr_leak["stop"] is not None, "first": leaky[0], "last": leaky[-1]},
        "unsloth_rule": {"limit": sc["unsloth_rule_train_loss"], "smooth_first_step": below + 1 if below is not None else None,
                         "noisy_first_step": noisy_below + 1 if noisy_below is not None else None},
    }


def e2_threshold(sc):
    out = []
    for th in [0.0] + sc["early_stopping"]["threshold_demo"]:
        t = M.early_stopping_trace(sc["hand_curve"], patience=sc["early_stopping"]["patience"], threshold=th)
        out.append({"threshold": th, "stop_eval": t["stop"] + 1 if t["stop"] is not None else None,
                    "best_eval": t["best"] + 1, "counters": [r["counter"] for r in t["rows"]]})
    return out


def e3_sampling(sc, graded):
    st = sc["sampling_toy"]
    res = {}
    for name in ("base", "ft_best"):
        g = [r["human"] for r in graded[name]]
        c = M.sampled_pass_counts(g, st["p_pass_if_greedy_pass"], st["p_pass_if_greedy_fail"], st["runs"], st["seed0"])
        res[name] = {"greedy": sum(g), "sampled": c, "min": min(c), "max": max(c), "mean": r4(sum(c) / len(c))}
    return res


def e4_scoring(graded):
    return {name: M.summarize(rows) for name, rows in graded.items()}


def e5_by_type(graded):
    bt = M.by_type(graded["base"], graded["ft_best"])
    ch01 = json.loads(CH01.read_text(encoding="utf-8"))
    pred = next(r for r in ch01["e3_compare"]["rows"] if r["combo"] == "F")
    return {"table": bt, "ch01_predicted_F_remaining_by_type": pred["remaining_by_type"],
            "ch01_predicted_F_remaining": pred["remaining"],
            "ft_best_remaining": 40 - sum(r["human"] for r in graded["ft_best"]),
            "last_vs_best": M.by_type(graded["ft_best"], graded["ft_last"])}


def e6_probe():
    pr = M.load("general_probe.json")["items"]
    base = [M.probe_pass(i["base"], i["expect"]) for i in pr]
    ft = [M.probe_pass(i["ft_best"], i["expect"]) for i in pr]
    bo, fo = M.discordant(base, ft)
    return {"n": len(pr), "base": sum(base), "ft_best": sum(ft), "base_only": bo, "ft_only": fo,
            "base_only_ids": [i["id"] for i, a, b in zip(pr, base, ft) if a and not b],
            "ft_only_ids": [i["id"] for i, a, b in zip(pr, base, ft) if b and not a],
            "pass_base": base, "pass_ft": ft}


def e7_judge(graded):
    hx = [r["human"] for r in graded["ft_best"]]
    hy = [r["human"] for r in graded["base"]]
    j = M.judge_both_orders(hx, hy)
    j["human_ft_only"] = sum(a and not b for a, b in zip(hx, hy))
    j["human_base_only"] = sum(b and not a for a, b in zip(hx, hy))
    j["human_ties"] = sum(a == b for a, b in zip(hx, hy))
    return j


def e8_chance(sc, graded, probe):
    bs = sc["bootstrap"]
    out = {}
    pairs = {
        "base_vs_ft_best": ([int(r["human"]) for r in graded["base"]], [int(r["human"]) for r in graded["ft_best"]]),
        "ft_last_vs_ft_best": ([int(r["human"]) for r in graded["ft_last"]], [int(r["human"]) for r in graded["ft_best"]]),
        "probe_base_vs_ft_best": ([int(x) for x in probe["pass_base"]], [int(x) for x in probe["pass_ft"]]),
    }
    for k, (a, b) in pairs.items():
        bt = M.paired_bootstrap(a, b, n_boot=bs["n_boot"], seed=bs["seed"], level=bs["level"])
        ao, bo = M.discordant(a, b)
        out[k] = {"a_pass": sum(a), "b_pass": sum(b), "a_only": ao, "b_only": bo,
                  "diff": r4(bt["diff"]), "lo": r4(bt["lo"]), "hi": r4(bt["hi"]), "contains_zero": bt["contains_zero"],
                  "sign_p": r4(M.sign_test(ao, bo)), "copies_needed": M.copies_needed(ao, bo, sc["alpha"])}
        if out[k]["copies_needed"]:
            out[k]["items_needed"] = out[k]["copies_needed"] * len(a)
    # 시드를 바꿔도 결론이 같은지
    a, b = pairs["ft_last_vs_ft_best"]
    out["seed_check_ft_last_vs_ft_best"] = [
        {"seed": s, **{kk: r4(v) if isinstance(v, float) else v
                       for kk, v in M.paired_bootstrap(a, b, n_boot=bs["n_boot"], seed=s).items()
                       if kk in ("lo", "hi", "contains_zero")}} for s in range(5)]
    return out


def main():
    sc = M.load("scenario.json")
    bank = M.load("answer_bank.json")
    sheet = M.load("score_sheet.json")["models"]
    inq = M.load_inquiries()
    graded = {name: M.grade_model(bank, sheet[name], inq) for name in ("base", "ft_best", "ft_last")}
    probe = e6_probe()
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "spec_file": M.SPEC_FILE.name, "checked": "2026-10-11", "eval_set": "ch01_when_finetune/data/inquiries.json",
                 "note": "손실 곡선과 채점 데이터는 모두 가상이다. 실제 학습 결과가 아니다."},
        "e0_hand": e0_hand(sc),
        "e1_curves": e1_curves(sc),
        "e2_threshold": e2_threshold(sc),
        "e3_sampling": e3_sampling(sc, graded),
        "e4_scoring": e4_scoring(graded),
        "e5_by_type": e5_by_type(graded),
        "e6_probe": {k: v for k, v in probe.items() if not k.startswith("pass_")},
        "e7_judge": e7_judge(graded),
        "e8_chance": e8_chance(sc, graded, probe),
        "examples": {q: {"reference": M.reference(bank, q),
                         **{n: next(r["answer"] for r in graded[n] if r["id"] == q) for n in graded}}
                     for q in ("Q01", "Q04", "Q14", "Q24", "Q40")},
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    r = main()
    for k in ("e0_hand", "e2_threshold", "e3_sampling", "e4_scoring", "e6_probe", "e7_judge", "e8_chance"):
        print(k, json.dumps(r[k], ensure_ascii=False)[:1500])
    c = r["e1_curves"]
    print({k: c[k] for k in c if k not in ("evals", "train_every_10", "trace")})
    for e in c["evals"]:
        print(e)
    print(json.dumps(r["e5_by_type"], ensure_ascii=False))
