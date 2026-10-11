"""내 데이터로 LLM 파인튜닝하기 9장 검증. `uv run pytest -q finetune/ch09_evaluate` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
GPU·네트워크·API 키 없이 돈다. torch·unsloth를 불러오지 않는다.
"""

import ast
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

import ft_ch09_eval as M

HERE = Path(__file__).parent
NB = HERE / "colab" / "ch09_evaluate_t4.ipynb"
RES = HERE / "results" / "ch09.json"


@pytest.fixture(scope="module")
def res():
    return json.loads(RES.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def graded():
    bank, sheet, inq = M.load("answer_bank.json"), M.load("score_sheet.json")["models"], M.load_inquiries()
    return {n: M.grade_model(bank, sheet[n], inq) for n in ("base", "ft_best", "ft_last")}


# ---------- 사양표 ----------

def test_spec_file_dated_and_sourced():
    s = M.load_specs()
    assert s["checked"] == "2026-10-11" and M.SPEC_FILE.name == "specs_2026-10-11.json"
    n_checked = 0

    def walk(d):
        nonlocal n_checked
        if isinstance(d, dict):
            if "checked" in d:
                assert d["checked"] == "2026-10-11"
                n_checked += 1
            for k, v in d.items():
                if k.startswith("url"):
                    assert v.startswith("https://"), (k, v)
                walk(v)
    walk(s)
    for k, v in s.items():
        if isinstance(v, dict):
            assert v.get("checked") == "2026-10-11", k
    assert s["unsloth_hyperparams"]["claim"] is True
    assert "[확인 필요]" in s["peft_disable_adapter"]["unsloth_interaction"]


def test_qwen_generation_config_matches_ch02_download():
    s = M.load_specs()["generation"]["qwen_instruct_generation_config"]
    g = json.loads((HERE.parent / "ch02_huggingface_map" / "data" / "qwen2.5-7b-instruct" / "generation_config.json").read_text())
    for k, v in s.items():
        assert g[k] == v


# ---------- 평가 세트 ----------

def test_eval_set_is_ch01_and_types_match_base_failures(graded):
    inq = M.load_inquiries()
    assert len(inq) == 40
    counts = {t: sum(1 for i in inq if i["failure"] == t) for t in M.TYPES}
    assert counts == {"K": 12, "F": 8, "T": 8, "I": 6, "R": 4, "OK": 2}
    # 가상 base 답의 사람 채점이 1장 이름표와 같은 곳에서 틀린다
    for r, it in zip(graded["base"], inq):
        assert r["human"] == (it["failure"] == "OK")
        h = r["human_detail"]
        if it["failure"] == "K" or it["failure"] == "R":
            assert not h["사실·계산"]
        if it["failure"] == "F":
            assert not (h["인사"] and h["맺음말"] and h["길이"])
        if it["failure"] == "T":
            assert not h["말투"]
        if it["failure"] == "I":
            assert not h["규칙"]


# ---------- 조기 종료 ----------

def test_early_stopping_hand_trace_matches_hf_rule():
    t = M.early_stopping_trace([1.20, 0.80, 0.62, 0.58, 0.61, 0.69], patience=2)
    assert [r["counter"] for r in t["rows"]] == [0, 0, 0, 0, 1, 2]
    assert t["stop"] == 5 and t["best"] == 3
    # 첫 평가는 best가 없어 카운터 0
    assert t["rows"][0]["prev_best"] is None


def test_threshold_counts_small_gain_as_no_gain_but_updates_best():
    t = M.early_stopping_trace([1.20, 0.80, 0.62, 0.58, 0.61, 0.69], patience=2, threshold=0.05)
    # 0.62 -> 0.58은 0.04만 좋아져 카운터가 오르지만, best는 0.58로 바뀐다
    assert [r["counter"] for r in t["rows"]] == [0, 0, 0, 1, 2]
    assert t["rows"][3]["best_after"] == 0.58 and t["stop"] == 4


def test_early_stopping_equal_value_is_not_improvement():
    t = M.early_stopping_trace([1.0, 1.0, 1.0], patience=2)
    assert [r["counter"] for r in t["rows"]] == [0, 1, 2]


def test_early_stopping_greater_is_better():
    t = M.early_stopping_trace([0.5, 0.6, 0.55, 0.58], patience=2, greater_is_better=True)
    assert t["best"] == 1 and t["stop"] == 3


def test_curves_deterministic_and_shape(res):
    sc = M.load("scenario.json")
    a = M.make_curves(sc["curve"], 34)
    b = M.make_curves(sc["curve"], 34)
    assert a == b and a["total_steps"] == 204 and len(a["evals"]) == 12
    c = res["e1_curves"]
    assert c["best_step"] == 85 and c["stop_step"] == 119 and c["every_epoch"]["best_step"] == 68
    assert c["ch08_notebook"]["steps"] == [34, 68, 102] and c["ch08_notebook"]["stopped"] is False
    assert c["leaky"]["stopped"] is False and c["leaky"]["min_step"] == 204
    # 매끈한 학습 손실은 계속 내려간다
    tr = [M.train_loss(s, sc["curve"]["train"]) for s in range(1, 205)]
    assert all(x > y for x, y in zip(tr, tr[1:]))


def test_curve_formula_by_hand():
    p = M.load("scenario.json")["curve"]["val"]
    v = 0.70 + 1.20 * math.exp(-85 / 26) + 0.000045 * 25 ** 2
    assert abs(M.val_loss(85, p) - v) < 1e-12


# ---------- 채점 ----------

def test_compose_and_matches():
    bank = M.load("answer_bank.json")
    ref = M.reference(bank, "Q14")
    pv = M.compose(bank, "Q14", "ref_punct")
    assert ref.startswith(bank["greeting"]) and ref.endswith(bank["signoff"])
    assert pv != ref and not M.exact_match(pv, ref) and M.normalized_match(pv, ref)
    alt = M.compose(bank, "Q01", "alt")
    assert not M.normalized_match(alt, M.reference(bank, "Q01"))


def test_rule_check_paraphrase_and_tone_blind_spots():
    bank = M.load("answer_bank.json")
    # 같은 뜻인데 낱말이 달라 규칙이 떨어뜨린다
    assert M.rule_check(bank, "Q01", M.compose(bank, "Q01", "alt"))["통과"] is False
    assert M.rule_check(bank, "Q04", M.compose(bank, "Q04", "alt"))["통과"] is False
    # 차가운 말투, 엉뚱한 답은 규칙이 못 본다
    assert M.rule_check(bank, "Q24", M.compose(bank, "Q24", "curt"))["통과"] is True
    assert M.rule_check(bank, "Q40", M.compose(bank, "Q40", "offtopic"))["통과"] is True
    # 금지 약속, 틀린 값, 빠진 인사는 잡는다
    assert M.rule_check(bank, "Q32", M.compose(bank, "Q32", "forbidden"))["통과"] is False
    assert M.rule_check(bank, "Q09", M.compose(bank, "Q09", "wrong"))["통과"] is False
    assert M.rule_check(bank, "Q13", M.compose(bank, "Q13", "ref", greet=False))["통과"] is False
    assert M.rule_check(bank, "Q20", M.compose(bank, "Q20", "long"))["통과"] is False


def test_every_ref_passes_rules_and_every_wrong_or_forbidden_fails():
    bank = M.load("answer_bank.json")
    for qid, it in bank["items"].items():
        assert M.rule_check(bank, qid, M.reference(bank, qid))["통과"], qid
        assert M.rule_check(bank, qid, M.compose(bank, qid, "ref_punct"))["통과"], qid
        for v in ("wrong", "forbidden"):
            if v in it:
                assert not M.rule_check(bank, qid, M.compose(bank, qid, v))["통과"], (qid, v)


def test_policy_values_match_ch01_policies():
    pol = json.loads((HERE.parent / "ch01_when_finetune" / "data" / "policies_v1.json").read_text(encoding="utf-8"))
    text = " ".join(d["text"] for d in pol["docs"])
    for v in ["7일", "3,000원", "3만 원", "당일 출고", "2~3일", "1회 무료", "6,000원", "1년", "함께 쓸 수 없어요", "무상 수리", "오후 6시"]:
        assert v in text


def test_r_answers_by_arithmetic():
    total = 28000 - 3000
    assert total < 30000 and total + 3000 == 28000          # Q35
    assert 120000 * 1 // 100 == 1200                        # Q36
    assert 0 + 6000 == 6000                                 # Q38: 사이즈 1회 무료 + 변심 6,000
    # Q37: 월요일 출고 + 1~2일 + 2~3일 = 3~5일 뒤 -> 목~토
    days = ["월", "화", "수", "목", "금", "토", "일"]
    assert days[0 + 3] == "목" and days[0 + 5] == "토"


def test_scoring_summary(res, graded):
    s = {n: M.summarize(r) for n, r in graded.items()}
    assert s["base"]["human"] == 2 and s["ft_best"]["human"] == 29 and s["ft_last"]["human"] == 27
    assert s["ft_best"]["exact"] == 11 and s["ft_best"]["normalized"] == 23 and s["ft_best"]["rule"] == 30
    assert s["ft_best"]["rule_false_pass"] == ["Q24", "Q26", "Q40"]
    assert s["ft_best"]["rule_false_fail"] == ["Q01", "Q04"]
    assert s["base"]["rule"] == 10 and len(s["base"]["rule_false_pass"]) == 8
    b = s["ft_best"]
    assert (b["tp"], b["fp"], b["fn"], b["tn"]) == (27, 3, 2, 8)
    assert b["precision"] == round(27 / 30, 4) and b["recall"] == round(27 / 29, 4)
    for n in s:
        for k in ("exact", "normalized", "rule", "human"):
            assert res["e4_scoring"][n][k] == s[n][k]


def test_by_type_table(res):
    t = res["e5_by_type"]["table"]
    assert [t[k]["b_pass"] for k in M.TYPES] == [9, 8, 6, 5, 0, 1]
    assert sum(t[k]["b_pass"] for k in M.TYPES) == 29
    assert t["OK"]["broke"] == ["Q40"]
    assert res["e5_by_type"]["ch01_predicted_F_remaining"] == 7


def test_probe(res):
    p = res["e6_probe"]
    assert (p["base"], p["ft_best"], p["base_only"], p["ft_only"]) == (18, 15, 4, 1)


# ---------- 채점자 순서 ----------

def test_toy_judge_orders(res):
    j = res["e7_judge"]
    assert j["x_wins_x_first"] == 39 and j["x_wins_y_first"] == 28 and j["inconsistent"] == 11
    assert j["x_wins_x_first"] - j["x_wins_y_first"] == j["human_ties"]


# ---------- 우연인가 ----------

def test_sign_test_hand_values():
    assert M.sign_test(0, 2) == 0.5
    assert M.sign_test(1, 3) == 2 * (1 + 4) / 16
    assert M.sign_test(4, 1) == 2 * (1 + 5) / 32
    assert M.copies_needed(1, 3) == 5 and M.copies_needed(4, 1) == 3


def test_sign_test_equals_brute_force():
    # 엇갈린 n개를 동전으로 던진 모든 경우를 세어 같은지 본다
    for a, b in [(1, 3), (4, 1), (2, 7), (0, 5)]:
        n = a + b
        k = min(a, b)
        cnt = sum(1 for m in range(2 ** n) if min(bin(m).count("1"), n - bin(m).count("1")) <= k)
        assert abs(M.sign_test(a, b) - min(1.0, cnt / 2 ** n)) < 1e-12


def test_percentile_matches_numpy():
    rng = random.Random(1)
    xs = sorted(rng.random() for _ in range(997))
    for q in (2.5, 50, 97.5):
        assert abs(M.percentile(xs, q) - float(np.percentile(xs, q))) < 1e-12


def test_bootstrap_matches_numpy_reimplementation(res):
    bank, sheet, inq = M.load("answer_bank.json"), M.load("score_sheet.json")["models"], M.load_inquiries()
    a = [int(r["human"]) for r in M.grade_model(bank, sheet["ft_last"], inq)]
    b = [int(r["human"]) for r in M.grade_model(bank, sheet["ft_best"], inq)]
    bt = M.paired_bootstrap(a, b, n_boot=1000, seed=9)
    # 같은 위치를 numpy로 다시 계산
    rng = random.Random(9)
    idx = np.array([[rng.randrange(40) for _ in range(40)] for _ in range(1000)])
    d = (np.array(b)[idx] - np.array(a)[idx]).mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    assert abs(bt["lo"] - lo) < 1e-12 and abs(bt["hi"] - hi) < 1e-12
    e = res["e8_chance"]
    assert e["ft_last_vs_ft_best"]["contains_zero"] is True
    assert e["base_vs_ft_best"]["contains_zero"] is False
    assert e["probe_base_vs_ft_best"]["contains_zero"] is True
    assert all(r["contains_zero"] for r in e["seed_check_ft_last_vs_ft_best"])


def test_results_deterministic(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("ch09_experiments", HERE / "experiments.py")
    mod = importlib.util.module_from_spec(spec)
    before = RES.read_bytes()
    spec.loader.exec_module(mod)
    mod.main()
    assert RES.read_bytes() == before


# ---------- 기록 양식과 노트북 ----------

def test_checks_csv():
    rows = list(csv.reader((HERE / "checks" / "ch09.csv").open(encoding="utf-8")))
    assert len(rows) >= 2 and "(예시·모형값)" in rows[1][0]
    assert all(len(r) == len(rows[0]) for r in rows)


def _py(src):
    """셸 명령(!, %) 줄은 들여쓰기를 지킨 채 pass로 바꿔 파이썬으로 읽는다."""
    out = []
    for l in src.splitlines():
        st = l.lstrip()
        out.append(l[: len(l) - len(st)] + "pass" if st.startswith(("!", "%")) else l)
    return "\n".join(out)


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def test_notebook_outputs_empty_and_parses():
    nb = _nb()
    assert nb["metadata"]["colab"]["gpuType"] == "T4"
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert c["outputs"] == [] and c["execution_count"] is None
            src = "".join(c["source"])
            ast.parse(_py(src))


def _nb_literal(name):
    for c in _nb()["cells"]:
        if c["cell_type"] != "code":
            continue
        tree = ast.parse(_py("".join(c["source"])))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == name for t in node.targets):
                return ast.literal_eval(node.value)
    raise KeyError(name)


def test_notebook_embeds_same_eval_set_rules_and_settings():
    inq = M.load_inquiries()
    assert _nb_literal("EVAL_SET") == [[i["id"], i["failure"], i["text"]] for i in inq]
    bank = M.load("answer_bank.json")
    assert _nb_literal("RULES") == {q: {"must": it.get("must", []), "forbid": it.get("forbid", [])} for q, it in bank["items"].items()}
    assert _nb_literal("GREETING") == bank["greeting"] and _nb_literal("SIGNOFF") == bank["signoff"]
    probe = M.load("general_probe.json")["items"]
    assert _nb_literal("PROBE_SET") == [[p["id"], p["q"], p["expect"]] for p in probe]
    sc = M.load("scenario.json")
    gen = _nb_literal("GEN")
    assert gen["do_sample"] is False and gen["max_new_tokens"] == sc["generation"]["max_new_tokens"]
    assert _nb_literal("SYSTEM") == sc["generation"]["system"]
    assert _nb_literal("PROBE_SYSTEM") == sc["generation"]["probe_system"] != sc["generation"]["system"]
    # 8장 노트북과 같은 system 문장
    nb8 = json.loads((HERE.parent / "ch08_unsloth_colab" / "colab" / "ch08_unsloth_qlora_t4.ipynb").read_text(encoding="utf-8"))
    assert any(sc["generation"]["system"] in "".join(c["source"]) for c in nb8["cells"])
    assert _nb_literal("PATIENCE") == sc["early_stopping"]["patience"]
    assert _nb_literal("N_BOOT") == sc["bootstrap"]["n_boot"]


def test_notebook_checks_row_has_header_column_count():
    """13번 칸이 찍는 한 줄을 가짜 값으로 돌려, checks/ch09.csv 머리줄과 열 수가 같은지 본다."""
    import contextlib
    import io
    import types
    cells = ["".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "code"]
    c12 = next(c for c in cells if c.startswith("# 12."))
    c13 = next(c for c in cells if c.startswith("# 13."))
    env = {"N_BOOT": 50, "BOOT_SEED": 9, "random": random, "math": math, "csv": csv,
           "hb": [0, 1] * 20, "hf": [1, 1] * 20, "probe_base": [True] * 20, "probe_ft": [True] * 19 + [False],
           "GEN": {"max_new_tokens": 256}, "eval_log": [(17, 1.0), (34, 0.9)], "best_i": 1, "stop_i": None,
           "unsloth": types.SimpleNamespace(__version__="x"),
           "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_name=lambda i: "T4"))}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(c12, env)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(c13, env)
    row = next(csv.reader([buf.getvalue().strip()]))
    header = next(csv.reader((HERE / "checks" / "ch09.csv").open(encoding="utf-8")))
    assert len(row) == len(header)
