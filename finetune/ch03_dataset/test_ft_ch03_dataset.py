"""파인튜닝 트랙 3장 검증. `uv run pytest -q finetune/ch03_dataset` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산하고,
results/ch03.json 의 숫자와도 맞춰 본다. GPU·네트워크·API 키 없이 돈다.
"""

import json
from pathlib import Path

import ft_ch03_clean as C
import ft_ch03_template as T

HERE = Path(__file__).parent
RES = json.loads((HERE / "results" / "ch03.json").read_text(encoding="utf-8"))
# 손계산용 대화 한 건. system, user, assistant 메시지 각각 한 개
HAND = [{"role": "system", "content": "친절한 상담원"},
        {"role": "user", "content": "포장이 찢어졌어요"},
        {"role": "assistant", "content": "속상하셨죠. 바로 새로 보내 드릴게요."}]


# ---------- 출처 파일 ----------

def test_spec_file_has_date_and_sources():
    s = T.load_specs()
    assert s["checked"] == "2026-10-11" and T.SPEC_FILE.name == "specs_2026-10-11.json"
    q = s["qwen2_5_instruct_template"]
    assert q["url"].startswith("https://huggingface.co/") and len(q["sha"]) == 40
    assert q["im_start"] == "<|im_start|>" and q["im_end"] == "<|im_end|>"
    assert s["trl"]["assistant_only_loss_default"] is False and s["trl"]["completion_only_loss_default"] is None
    assert s["pytorch_cross_entropy"]["ignore_index_default"] == -100 == T.IGNORE
    assert s["qwen2_5_base_tokenizer"]["eos_token"] == s["qwen2_5_base_tokenizer"]["pad_token"]


# ---------- 대화 템플릿 ----------

def test_render_matches_qwen_message_line():
    out = T.render(HAND)
    assert out == ("<|im_start|>system\n친절한 상담원<|im_end|>\n"
                   "<|im_start|>user\n포장이 찢어졌어요<|im_end|>\n"
                   "<|im_start|>assistant\n속상하셨죠. 바로 새로 보내 드릴게요.<|im_end|>\n")


def test_default_system_inserted_when_missing():
    out = T.render(HAND[1:])
    assert out.startswith("<|im_start|>system\nYou are Qwen, created by Alibaba Cloud.")


def test_generation_prompt_only_when_asked():
    m = HAND[:2]
    assert not T.render(m).endswith("<|im_start|>assistant\n")
    assert T.render(m, add_generation_prompt=True).endswith("<|im_start|>assistant\n")


def test_segments_join_to_render():
    assert "".join(s for s, _ in T.segments(HAND)) == T.render(HAND)


# ---------- 마스킹 ----------

def test_hand_counts():
    c = T.counts(HAND)
    # 가르치기용 세기 기준 52토큰 중 24개만 학습하고 28개는 무시. 실제 Qwen 토큰 수가 아니다
    assert c == {"total": 52, "trained": 24, "ignored": 28}
    # 답 21글자(띄어쓰기·마침표 포함) + 앞 줄바꿈 + <|im_end|> + 뒤 줄바꿈 = 24
    assert len("속상하셨죠. 바로 새로 보내 드릴게요.") + 3 == 24
    assert RES["e0_hand"]["total"] == 52 and RES["e0_hand"]["trained"] == 24


def test_mask_boundaries():
    toks, mask = T.tokens_and_mask(HAND)
    i = mask.index(True)
    # 학습 구간 바로 앞 두 토큰이 <|im_start|>, assistant
    assert toks[i - 2:i] == ["<|im_start|>", "assistant"]      # 머리말은 학습하지 않는다
    assert toks[i] == "\n" and toks[i + 1] == "속"               # 답 첫 글자는 학습한다
    # 답의 끝 표시와 뒤 줄바꿈도 학습한다(모델이 멈추는 법을 배워야 하므로)
    assert toks[-2:] == ["<|im_end|>", "\n"] and mask[-2:] == [True, True]   # 멈추는 표시도 학습한다
    assert not any(mask[:i])


def test_labels_same_length_no_shift():
    toks, mask = T.tokens_and_mask(HAND)
    ids = list(range(len(toks)))
    lab = T.labels(ids, mask)
    assert len(lab) == len(ids)
    # 마스크가 참이면 라벨이 -100이 아니고, 거짓이면 -100이어야 한다(배타적 논리)
    assert all((l == T.IGNORE) != m for l, m in zip(lab, mask))
    assert all(l == i for l, i, m in zip(lab, ids, mask) if m)  # 자기 자리 값 그대로(밀지 않음)


def test_masked_mean_ignores_prompt():
    loss = [5.0, 5.0, 1.0, 3.0]
    # (1.0 + 3.0) / 2 = 2.0. 앞의 5.0 두 개는 제외
    assert T.masked_mean(loss, [False, False, True, True]) == 2.0


def test_pad_equals_eos_trap():
    eot = T.EOT
    a, b = T.toy_tokens("고마워요" + eot), T.toy_tokens("잘 받았어요 감사해요" + eot)
    padded = T.pad_batch([a, b], eot)
    by_val = T.labels_mask_by_pad_value(padded, eot)
    by_len = T.labels_mask_by_length(padded, [len(a), len(b)])
    # 값으로 가리면 진짜 끝 표시까지 모두 지워진다. 길이로 가리면 두 문장 모두 끝 표시가 하나씩 남는다(합 2)
    assert sum(r.count(eot) for r in by_val) == 0       # 끝 표시가 모두 가려진다
    assert sum(r.count(eot) for r in by_len) == 2       # 길이로 가리면 끝 표시가 남는다


# 정제를 마친 데이터 전체에서 학습 토큰이 차지하는 몫을 처음부터 다시 계산해 results 값과 대조
def test_dataset_share_recomputed():
    items = C.load_raw()
    s1, _ = C.drop_empty(items)
    s2, _ = C.drop_volatile(s1)
    s3, _ = C.drop_exact(s2)
    s4, _, _ = C.drop_near(s3, 0.8)
    s5, _ = C.drop_leaks(s4, C.load_eval(), 0.5)
    sys_ = RES["meta"]["system_prompt"]
    tot = tr = 0
    for it in s5:
        c = T.counts([{"role": "system", "content": sys_}, {"role": "user", "content": it["q"]},
                      {"role": "assistant", "content": it["a"]}])
        tot += c["total"]
        tr += c["trained"]
    e = RES["e2_share"]["our_system"]
    assert (tot, tr) == (e["total"], e["trained"]) and round(tr / tot, 4) == e["share"]


def test_pipeline_recomputed_matches_results():
    """results 파일이 낡아도 통과하지 않게, 모듈로 처음부터 다시 돌려 맞춘다."""
    raw = C.load_raw()
    ev = C.load_eval()
    s1, _ = C.drop_empty(raw)
    s2, _ = C.drop_volatile(s1)
    s3, _ = C.drop_exact(s2)
    s4, _, _ = C.drop_near(s3, RES["meta"]["near_threshold"])
    s5, leaks = C.drop_leaks(s4, ev, RES["meta"]["leak_threshold"])
    # results 파일이 낡아도 통과하지 않게 정제 단계별 남은 개수를 모듈로 다시 계산해 비교
    assert [len(x) for x in (raw, s1, s2, s3, s4, s5)] == [r["left"] for r in RES["e4_funnel"]["rows"]]
    assert sorted((d["id"], d["eval_id"]) for d in leaks) == sorted((a, b) for a, b, _ in RES["e4_funnel"]["leak_ids"])
    val_n = round(RES["meta"]["val_ratio"] * len(s5))
    tr, va = C.split_seeded(s5, val_n, RES["meta"]["seed"])
    assert [v["id"] for v in va] == RES["e7_clean_split"]["val_ids"]
    assert sum(t["twin"] for t in C.cross_twins(tr, va, 0.8)) == 0 and C.fake_memorizer(tr, va)["passed"] == 0
    # 지우기 전에 나눈 시드 3
    val_n2 = round(RES["meta"]["val_ratio"] * len(s2))
    # 중복을 지우기 전(s2)에 나누면 닮은꼴이 검증 쪽에 남아 가짜 모델이 5건을 맞힌다(결과 파일의 숫자와 같은지 확인)
    a, b = C.split_seeded(s2, val_n2, RES["meta"]["seed"])
    assert C.fake_memorizer(a, b)["passed"] == RES["e8_leaky_split"]["seed_memorizer_passed"] == 5
    # 해시 나누기: 새 항목을 더해도 원래 항목은 제자리
    t1, v1 = C.split_hash(s5, 0.1)
    extra = [{"id": f"N{i:03d}", "q": "새 문의", "a": "새 답", "planted": None} for i in range(1, 21)]
    t2, v2 = C.split_hash(s5 + extra, 0.1)
    # 해시로 나누면 새 항목 20개를 더해도 원래 검증 항목이 그대로
    assert {v["id"] for v in v1} == {v["id"] for v in v2 if not v["id"].startswith("N")}


# ---------- 닮음과 정제 ----------

def test_pieces_reuse_ch01_rule():
    assert C.pieces("앱이 자꾸 꺼져요") == {"앱이", "자꾸", "꺼져", "져요"}
    assert C.pieces("주문번호 A1023 확인") == C.pieces("주문번호 B5521 확인")


def test_hand_jaccard():
    d = C.jaccard_detail("앱이 자꾸 꺼져요", "앱이 계속 꺼져요")
    # 교집합 3(앱이, 꺼져, 져요), 합집합 5이므로 자카드 3/5 = 0.6
    assert (d["inter"], d["union"]) == (3, 5) and d["j"] == 0.6


def test_normalize_catches_format_variants():
    assert C.normalize(" 충전이 안 돼요. ") == C.normalize("충전이  안 돼요")
    assert C.normalize("상품이 왔어요 ㅠㅠ") == C.normalize("상품이 왔어요!")


def test_volatile_rule_keeps_material_answer():
    # '면 100%'처럼 정책 낱말 없이 숫자만 있는 답은 남기고, '배송비 3,000원'은 자주 바뀌는 값으로 뺀다
    assert not C.is_volatile("네, 겉감은 면 100%예요.")
    assert C.is_volatile("배송비 3,000원이에요.")


def test_funnel_numbers():
    rows = {r["step"]: r["left"] for r in RES["e4_funnel"]["rows"]}
    # 정제 깔때기 단계별 남은 건수. 원본 83건에서 51건이 남는다(results 파일 기준)
    assert rows == {"raw": 83, "drop_empty": 81, "drop_volatile": 76, "drop_exact": 59, "drop_near": 55, "drop_leak": 51}


def test_detector_finds_every_planted_problem():
    c = RES["e10_crosscheck"]
    assert c["planted_missed"] == [] and c["removed_not_planted"] == []
    assert c["union_find_equals_bfs"] and c["jaccard_symmetric"]


def test_leaks_against_ch01_eval_set():
    ev = C.load_eval()
    assert len(ev) == 40
    flagged = {r["id"]: r["eval_id"] for r in RES["e6_leak"]["flagged"]}
    # 학습 후보 4건(R078~R081)이 1장 평가 문의와 겹친다고 표시된다(자카드 0.5 기준)
    assert flagged == {"R078": "Q21", "R079": "Q14", "R080": "Q20", "R081": "Q18"}


# ---------- 나누기 ----------

def test_split_is_deterministic_and_disjoint():
    items = C.load_raw()[:40]
    a1, b1 = C.split_seeded(items, 4, 3)
    a2, b2 = C.split_seeded(items, 4, 3)
    assert [x["id"] for x in b1] == [x["id"] for x in b2]
    assert not ({x["id"] for x in a1} & {x["id"] for x in b1}) and len(a1) + len(b1) == 40


def test_clean_split_has_no_twins_and_memorizer_scores_zero():
    e = RES["e7_clean_split"]
    # 정제 후 나누면 검증 쪽 닮은꼴 0개, 외우기 가짜 모델의 통과 0건. 시드 1000개 최댓값도 0
    assert e["twins"] == 0 and e["memorizer"]["passed"] == 0 and e["memorizer_1000_seeds_max"] == 0


def test_leaky_split_inflates_fake_score():
    e = RES["e8_leaky_split"]
    assert e["twins_mean"] > 3 and e["memorizer_mean"] == e["twins_mean"]
    assert e["memorizer_equals_twins_every_seed"] is True
    assert sum(e["twin_count_distribution"].values()) == 1000 == sum(e["memorizer_count_distribution"].values())
    # 시드 1000개 중 가짜 모델이 하나라도 맞힌 시드가 984개(결과 파일의 값과 같다는 확인)
    assert 1000 - e["memorizer_zero_seeds"] == 984


def test_hash_split_keeps_old_items_in_place():
    assert RES["e9_hash_stability"]["hash_old_items_same_side"] is True
    assert RES["e9_hash_stability"]["seeded_old_val_now_train"] > 0
