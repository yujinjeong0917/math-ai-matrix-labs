"""내 데이터로 LLM 파인튜닝하기 3장 실험. `uv run python finetune/ch03_dataset/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 대화 한 건을 템플릿에 넣고, 학습하는 토큰과 무시하는 토큰을 센다(가르치기용 세기)
E1 system 메시지를 빼면: Qwen2.5 템플릿이 영어 기본 문장을 넣는다(글만 확인, 수는 세지 않음)
E2 데이터 전체에서 답 토큰의 몫(우리 system 문장을 넣을 때, 가르치기용 세기)
E3 패드 = 끝 표시일 때 '패드 값으로 가리기'가 끝 표시까지 지우는지
E4 정제 깔때기: 빈 답 → 자주 바뀌는 정책 값 → 똑같은 중복 → 거의 중복 → 평가 세트 누수
E5 거의 중복: 손 예시, 찾은 쌍, 기준값 바꿔 보기
E6 누수 검사: 1장 평가 문의 40건과 겹치는 학습 후보, 기준값 바꿔 보기
E7 깨끗한 데이터로 나누기와 '외워서 답하는 가짜 모델'
E8 중복을 지우기 전에 나누면: 시드 1,000개로 센 검증 쪽 닮은꼴 수와 가짜 모델 점수
E9 해시로 나누기: 데이터가 늘어도 이미 있던 항목의 자리가 그대로인지
E10 다른 방법으로 대 보기: 유니온 파인드 묶음과 전수 비교, 자카드 대칭

표준 라이브러리만 쓴다. 무작위는 random.Random(seed)와 hashlib만 쓰고, 결과는 매번 같다.
"""

import json
import platform
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ft_ch03_clean as C  # noqa: E402
import ft_ch03_template as T  # noqa: E402

OUT = HERE / "results" / "ch03.json"
NEAR_T = 0.8
LEAK_T = 0.5
VAL_RATIO = 0.1
SEED = 3
SYSTEM = "우리 쇼핑몰 상담원이에요. 따뜻하고 짧게 답해요."
HAND = [{"role": "system", "content": "친절한 상담원"},
        {"role": "user", "content": "포장이 찢어졌어요"},
        {"role": "assistant", "content": "속상하셨죠. 바로 새로 보내 드릴게요."}]


def chat(it, system=SYSTEM):
    m = [{"role": "user", "content": it["q"]}, {"role": "assistant", "content": it["a"]}]
    return ([{"role": "system", "content": system}] if system else []) + m


def pipeline():
    raw = C.load_raw()
    ev = C.load_eval()
    s1, d_empty = C.drop_empty(raw)
    s2, d_vol = C.drop_volatile(s1)
    s3, d_exact = C.drop_exact(s2)
    s4, d_near, near_pairs = C.drop_near(s3, NEAR_T)
    s5, d_leak = C.drop_leaks(s4, ev, LEAK_T)
    return dict(raw=raw, ev=ev, s1=s1, s2=s2, s3=s3, s4=s4, s5=s5, d_empty=d_empty, d_vol=d_vol,
                d_exact=d_exact, d_near=d_near, near_pairs=near_pairs, d_leak=d_leak)


def e0_hand():
    toks, mask = T.tokens_and_mask(HAND)
    c = T.counts(HAND)
    seg = [{"text": s, "trained": tr, "toy_tokens": len(T.toy_tokens(s))} for s, tr in T.segments(HAND)]
    return {"messages": HAND, "rendered": T.render(HAND), "segments": seg, **c,
            "trained_share": round(c["trained"] / c["total"], 4),
            "first_trained": toks[mask.index(True)], "last_two_trained": toks[-2:],
            "labels_preview": [t if m else T.IGNORE for t, m in zip(toks, mask)][24:32]}


def e1_default_system():
    m = HAND[1:]
    return {"rendered": T.render(m), "default_system": T.load_specs()["qwen2_5_instruct_template"]["default_system"],
            "with_gen_prompt_tail": T.render(m[:1], add_generation_prompt=True)[-len("<|im_start|>assistant\n"):]}


def e2_share(p):
    out = {}
    # system을 뺀 경우(영어 기본 문장)는 세지 않는다. 가르치기용 세기는 영어 글자도 하나씩 세서 실제와 방향까지 어긋날 수 있다.
    for name, sys_ in (("our_system", SYSTEM),):
        tot = tr = 0
        for it in p["s5"]:
            c = T.counts(chat(it, sys_))
            tot += c["total"]
            tr += c["trained"]
        out[name] = {"pairs": len(p["s5"]), "total": tot, "trained": tr, "share": round(tr / tot, 4)}
    return out


def e3_pad_trap():
    eot = T.EOT
    a = T.toy_tokens("고마워요" + eot)
    b = T.toy_tokens("잘 받았어요 감사해요" + eot)
    padded = T.pad_batch([a, b], eot)
    lens = [len(a), len(b)]
    by_val = T.labels_mask_by_pad_value(padded, eot)
    by_len = T.labels_mask_by_length(padded, lens)
    kept = lambda rows: sum(row.count(eot) for row in rows)
    return {"lengths": lens, "padded_len": len(padded[0]), "pad_token": eot, "eos_token_base": eot,
            "eos_kept_by_value": kept(by_val), "eos_kept_by_length": kept(by_len), "examples": 2}


def e4_funnel(p):
    steps = [("raw", p["raw"]), ("drop_empty", p["s1"]), ("drop_volatile", p["s2"]), ("drop_exact", p["s3"]),
             ("drop_near", p["s4"]), ("drop_leak", p["s5"])]
    rows = []
    prev = None
    for name, items in steps:
        rows.append({"step": name, "left": len(items), "removed": 0 if prev is None else prev - len(items)})
        prev = len(items)
    exact_kinds = {}
    for d in p["d_exact"]:
        k = (d["planted"] or "none").split(":")[0]
        exact_kinds[k] = exact_kinds.get(k, 0) + 1
    return {"rows": rows, "volatile_ids": [d["id"] for d in p["d_vol"]], "empty_ids": [d["id"] for d in p["d_empty"]],
            "exact_by_kind": dict(sorted(exact_kinds.items())),
            "near_ids": [[d["id"], d["near_of"]] for d in p["d_near"]],
            "leak_ids": [[d["id"], d["eval_id"], d["score"]] for d in p["d_leak"]]}


def e5_near(p):
    hand = C.jaccard_detail("앱이 자꾸 꺼져요", "앱이 계속 꺼져요")
    whole = round(C.jaccard(C.pair_text(next(x for x in p["raw"] if x["id"] == "R033")),
                            C.pair_text(next(x for x in p["raw"] if x["id"] == "R077"))), 4)
    allp = sorted(C.near_pairs(p["s3"], 0.0), key=lambda t: (-t[2], t[0], t[1]))
    top = allp[:6]
    sweep = []
    for t in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        sweep.append({"threshold": t, "pairs": sum(s >= t for _, _, s in allp)})
    found = sorted(C.near_pairs(p["s3"], NEAR_T))
    return {"hand": hand, "R033_R077_whole_pair": whole, "top_pairs": top, "sweep": sweep, "found": found,
            "gap": {"lowest_found": min(s for _, _, s in found), "highest_not_found": max(s for _, _, s in allp if s < NEAR_T)}}


def e6_leak(p):
    sc = sorted(C.leak_scores(p["s4"], p["ev"]), key=lambda r: (-r["score"], r["id"]))
    hand = C.jaccard_detail("주문 취소하고 싶어요", next(e["text"] for e in p["ev"] if e["id"] == "Q20"))
    sweep = [{"threshold": t, "flagged": sum(r["score"] >= t for r in sc)} for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)]
    return {"eval_items": len(p["ev"]), "top": sc[:6], "hand_Q20": hand, "sweep": sweep,
            "flagged": [r for r in sc if r["score"] >= LEAK_T],
            "highest_not_flagged": max(r["score"] for r in sc if r["score"] < LEAK_T)}


def e7_clean_split(p):
    items = p["s5"]
    val_n = round(VAL_RATIO * len(items))
    tr, va = C.split_seeded(items, val_n, SEED)
    twins = C.cross_twins(tr, va, NEAR_T)
    mem = C.fake_memorizer(tr, va)
    many = []
    for seed in range(1000):
        a, b = C.split_seeded(items, val_n, seed)
        many.append(C.fake_memorizer(a, b)["passed"])
    return {"pairs": len(items), "train": len(tr), "val": len(va), "seed": SEED, "val_ids": [v["id"] for v in va],
            "twins": sum(t["twin"] for t in twins), "max_val_train_sim": max(t["score"] for t in twins),
            "memorizer": mem, "memorizer_1000_seeds_mean": statistics.mean(many), "memorizer_1000_seeds_max": max(many)}


def e8_leaky_split(p):
    items = p["s2"]          # 빈 답과 정책 값만 지우고, 중복은 그대로 둔 채 나눈다
    val_n = round(VAL_RATIO * len(items))
    tw, mp, dist, mdist = [], [], {}, {}
    for seed in range(1000):
        a, b = C.split_seeded(items, val_n, seed)
        t = sum(r["twin"] for r in C.cross_twins(a, b, NEAR_T))
        m = C.fake_memorizer(a, b)["passed"]
        tw.append(t)
        mp.append(m)
        dist[t] = dist.get(t, 0) + 1
        mdist[m] = mdist.get(m, 0) + 1
    a, b = C.split_seeded(items, val_n, SEED)
    shown = C.fake_memorizer(a, b)
    return {"pairs": len(items), "val": val_n, "seeds": 1000, "twins_mean": round(statistics.mean(tw), 3),
            "memorizer_mean": round(statistics.mean(mp), 3), "twins_min": min(tw), "twins_max": max(tw),
            "seeds_with_zero_twins": dist.get(0, 0), "twin_count_distribution": dict(sorted(dist.items())),
            "memorizer_equals_twins_every_seed": tw == mp, "memorizer_zero_seeds": mdist.get(0, 0),
            "memorizer_count_distribution": dict(sorted(mdist.items())),
            "seed": SEED, "seed_twins": sum(r["twin"] for r in C.cross_twins(a, b, NEAR_T)), "seed_memorizer_passed": shown["passed"],
            "seed_rows": shown["rows"]}


def e9_hash_stability(p):
    items = p["s5"]
    tr1, va1 = C.split_hash(items, VAL_RATIO)
    extra = [{"id": f"N{i:03d}", "q": f"새 문의 {i}", "a": f"새 답 {i}", "planted": None} for i in range(1, 21)]
    tr2, va2 = C.split_hash(items + extra, VAL_RATIO)
    old_val = {v["id"] for v in va1}
    new_val_old = {v["id"] for v in va2 if not v["id"].startswith("N")}
    # 비교: 시드로 섞어 나누면 항목이 늘 때 자리가 바뀐다
    s_tr1, s_va1 = C.split_seeded(items, round(VAL_RATIO * len(items)), SEED)
    s_tr2, s_va2 = C.split_seeded(items + extra, round(VAL_RATIO * len(items + extra)), SEED)
    s_old = {v["id"] for v in s_va1}
    s_new = {v["id"] for v in s_va2 if not v["id"].startswith("N")}
    moved_to_train = sorted(s_old - s_new)
    return {"hash_val_before": len(va1), "hash_val_after": len(va2), "hash_old_items_same_side": old_val == new_val_old,
            "seeded_val_before": len(s_va1), "seeded_old_val_now_train": len(moved_to_train),
            "seeded_old_train_now_val": len(s_new - s_old)}


def e10_crosscheck(p):
    items = p["s3"]
    pairs = C.near_pairs(items, NEAR_T)
    groups = C.clusters([it["id"] for it in items], pairs)
    # 전수 비교로 연결 요소를 다시 구한다(너비 우선)
    adj = {it["id"]: set() for it in items}
    for a, b, _ in pairs:
        adj[a].add(b)
        adj[b].add(a)
    seen, comps = set(), []
    for i in sorted(adj):
        if i in seen:
            continue
        comp, stack = [], [i]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.append(x)
            stack.extend(sorted(adj[x] - seen))
        comps.append(sorted(comp))
    sym = all(abs(C.jaccard(C.pair_text(a), C.pair_text(b)) - C.jaccard(C.pair_text(b), C.pair_text(a))) < 1e-12
              for a in items[:20] for b in items[:20])
    planted = {it["id"]: it["planted"] for it in p["raw"]}
    # 심어 둔 문제를 검사기가 다 찾았나(검사기는 planted 칸을 보지 않는다)
    removed = {d["id"]: kind for kind in ("d_empty", "d_vol", "d_exact", "d_near", "d_leak") for d in p[kind]}
    missed = sorted(i for i, pl in planted.items() if pl and i not in removed)
    false_pos = sorted(i for i in removed if not planted[i])
    return {"union_find_equals_bfs": sorted(groups) == sorted(comps), "groups_with_2plus": sum(len(g) > 1 for g in groups),
            "jaccard_symmetric": sym, "planted_total": sum(1 for v in planted.values() if v),
            "planted_missed": missed, "removed_not_planted": false_pos}


def main():
    p = pipeline()
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "spec_file": T.SPEC_FILE.name, "checked": "2026-10-11", "near_threshold": NEAR_T, "leak_threshold": LEAK_T,
                 "val_ratio": VAL_RATIO, "seed": SEED, "system_prompt": SYSTEM,
                 "toy_tokenizer": "특수 표시·역할 이름·줄바꿈은 1개, 나머지 글자 하나가 1개(가르치기용, 실제 토크나이저와 다름)"},
        "e0_hand": e0_hand(), "e1_default_system": e1_default_system(), "e2_share": e2_share(p),
        "e3_pad_trap": e3_pad_trap(), "e4_funnel": e4_funnel(p), "e5_near": e5_near(p), "e6_leak": e6_leak(p),
        "e7_clean_split": e7_clean_split(p), "e8_leaky_split": e8_leaky_split(p), "e9_hash_stability": e9_hash_stability(p),
        "e10_crosscheck": e10_crosscheck(p),
    }
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
