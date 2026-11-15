"""AI 도구 실전 4장 검증. `uv run pytest -q tools/ch04_skills` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
네트워크·API 키 없이 돈다.
"""

import json
from fractions import Fraction as F
from pathlib import Path

import pytest

import tools_ch04_budget as B
import tools_ch04_frontmatter as FM

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def skills():
    """스킬 6개를 한 번만 읽어 모든 테스트가 공유한다(module 범위)."""
    return FM.load_skills()


# ---------- 앞머리 파서 ----------

def test_parse_main_skill(skills):
    """대표 스킬의 칸 순서, 지도(metadata), 본문 시작, 딸린 파일 목록을 확인한다."""
    main = next(s for s in skills if s["dir"] == "weekly-inquiry-report")
    assert list(main["fm"].keys()) == ["name", "description", "metadata"]
    assert main["fm"]["name"] == "weekly-inquiry-report"
    assert main["fm"]["metadata"] == {"owner": "cs-team", "version": "1.3"}
    assert main["body"].lstrip().startswith("# 주간 고객 문의 보고서")
    assert "references/categories.md" in main["files"] and "scripts/count_by_type.py" in main["files"]


def test_all_example_skills_pass_spec_and_anthropic_rules(skills):
    """예시 스킬 6개가 두 규칙 묶음을 모두 통과한다(오류 목록이 비어 있음)."""
    assert len(skills) == 6
    for s in skills:
        assert FM.validate_spec(s["fm"], s["dir"]) == []
        assert FM.validate_anthropic(s["fm"]) == []


def test_body_references_exist(skills):
    """본문이 가리키는 파일이 실제로 있는지(끊어진 참조 방지)."""
    main = next(s for s in skills if s["dir"] == "weekly-inquiry-report")
    for p in ("scripts/count_by_type.py", "references/categories.md", "references/template.md"):
        assert f"`{p}`" in main["body"] and p in main["files"]


@pytest.mark.parametrize("name,ok", [("pdf-processing", True), ("a", True), ("a1-b2", True),
                                     ("PDF-Processing", False), ("-pdf", False), ("pdf-", False),
                                     ("pdf--processing", False), ("주간보고", False), ("a" * 64, True),
                                     ("a" * 65, False)])
def test_name_rule(name, ok):
    """이름 규칙 경계값. 64자는 통과, 65자는 실패. 한글은 a-z0-9 밖이라 실패."""
    errs = FM.validate_spec({"name": name, "description": "x"}, name)
    assert (errs == []) is ok


def test_description_limits():
    """description 경계: 1,024자 통과, 1,025자 실패, 없거나 공백뿐이면 실패."""
    assert FM.validate_spec({"name": "x", "description": "가" * 1024}) == []
    assert "description 1024자 초과" in FM.validate_spec({"name": "x", "description": "가" * 1025})
    assert "description 없음" in FM.validate_spec({"name": "x"})
    assert "description 없음" in FM.validate_spec({"name": "x", "description": "   "})


def test_dir_mismatch_and_anthropic_extras():
    """폴더 이름 불일치, 예약어, XML 태그. 마지막 줄은 두 규칙 묶음이 서로 다른 것을 검사한다는 확인."""
    assert "name이 폴더 이름과 다름" in FM.validate_spec({"name": "a", "description": "x"}, "b")
    assert FM.validate_anthropic({"name": "claude-report", "description": "x"}) == ["name에 예약어 'claude'"]
    assert FM.validate_anthropic({"name": "anthropic-x", "description": "x"}) == ["name에 예약어 'anthropic'"]
    assert FM.validate_anthropic({"name": "a", "description": "<b>x</b>"}) == ["description에 XML 태그"]
    # 사양 규칙만으로는 예약어를 막지 않는다
    assert FM.validate_spec({"name": "claude-report", "description": "x"}) == []


def test_parse_errors_and_quotes():
    """잘못된 앞머리는 오류, 따옴표는 벗기고 값 안의 ':'는 보존한다."""
    with pytest.raises(FM.FrontmatterError):
        FM.parse("name: x\n")
    with pytest.raises(FM.FrontmatterError):
        FM.parse("---\nname: x\n")
    with pytest.raises(FM.FrontmatterError):
        FM.parse("---\nname x\n---\n")
    fm, body = FM.parse("---\nname: 'a-b'\ndescription: \"설명: 콜론 포함\"\n---\n본문\n")
    assert fm == {"name": "a-b", "description": "설명: 콜론 포함"} and body == "본문\n"


def test_listing_truncation():
    """목록 한 줄은 description + 공백 + when_to_use를 1,536자에서 자른다. 1,000 + 1 + 1,000 > 1,536이라 잘린다."""
    fm = {"name": "x", "description": "a" * 1000, "when_to_use": "b" * 1000}
    line, cut = FM.listing_line(fm)
    assert cut and line == "x: " + ("a" * 1000 + " " + "b" * 1000)[:1536]
    assert FM.listing_line({"name": "y", "description": "짧아요"}) == ("y: 짧아요", False)


# ---------- 토큰 근사 ----------

def test_token_rule_hand():
    """토큰 근사 규칙을 손계산과 맞춘다. 올림(ceil)이 핵심."""
    assert B.tokens("") == 0
    # 기본 규칙: ASCII 4글자 = 1토큰, 올림. 5글자는 1.25 -> 2.
    assert B.tokens("abcd") == 1 and B.tokens("abcde") == 2
    # 한글은 기본 규칙에서 글자당 1토큰.
    assert B.tokens("가나다") == 3
    # 공백도 ASCII 글자로 센다.
    assert B.tokens("ab 가") == 2  # ASCII 3글자 -> 1, 한글 1글자 -> 1
    rule = {"ascii_chars_per_token": 4, "non_ascii_tokens_per_char": 0.6}
    assert B.tokens("가나다", rule) == 2  # 1.8 올림


def test_main_meta_hand(skills):
    """대표 스킬 목록 한 줄의 글자 수 145 중 ASCII 74, 나머지 71. 74/4 = 18.5 올림 19, 71 x 1 = 71, 합 90."""
    main = next(s for s in skills if s["dir"] == "weekly-inquiry-report")
    t = B.meta_text(main)
    n_ascii = sum(1 for ch in t if ord(ch) < 128)
    assert len(t) == 145 and n_ascii == 74
    assert B.tokens(t) == 19 + 71 == 90


def test_scripts_are_not_counted_as_references(skills):
    """scripts/ 아래는 실행 결과만 문맥에 들어가므로 읽는 파일 집계(refs)에서 빠진다."""
    sz = {x["name"]: x for x in B.sizes(skills)}
    assert sz["weekly-inquiry-report"]["scripts"] == ["scripts/count_by_type.py"]
    assert set(sz["weekly-inquiry-report"]["refs"]) == {"references/categories.md", "references/template.md"}


# ---------- 첫 화면 손계산과 식 ----------

def test_first_screen_hand():
    """N=5, 앞머리 100, 본문 2,000: 전부 10,500 대 단계적 2,500(76.2% 절감), N이 크면 한계 95.2%."""
    m, b = 100, 2000
    assert 5 * (m + b) == 10_500 and 5 * m + b == 2_500
    assert 1 * (m + b) == 1 * m + b == 2_100
    assert F(5 * m + b, 5 * (m + b)) == F(2500, 10500)
    assert round(float(100 * (1 - F(2500, 10500))), 1) == 76.2
    assert round(float(100 * (1 - F(m, m + b))), 1) == 95.2


def test_scale_closed_form():
    """문서 대략값(100, 5,000)으로 N=1, 10, 100. N이 커지면 단계적 값은 N x 100 + 5,000으로 목록 비용이 지배한다."""
    rows = B.scale(100, 5000, [1, 10, 100])
    assert [r["staged"] for r in rows] == [5100, 6000, 15000]
    assert [r["paste_all"] for r in rows] == [5100, 51000, 510000]
    assert rows[-1]["meta_only"] == 10000


# ---------- 한 주 ----------

def test_week_totals(skills):
    """한 주 합계: 전부 싣기 15 x 3,784 = 56,760, 단계적 16,244, 절감 71.4%. 기대값은 현재 data 파일의 토큰 근사에서 나온 값이라 데이터를 고치면 바뀐다."""
    res = B.week(skills)
    assert len(res["rows"]) == 15
    assert res["paste_all_total"] == 15 * 3784 == 56_760
    assert res["staged_total"] == 16_244
    assert round(100 * float(res["saving"]), 1) == 71.4
    r1 = res["rows"][0]
    # 첫 대화의 구성: 프로젝트 지침 245 + 목록 429 + 본문 412 + 딸린 파일 935 + 스크립트 출력 40(parts 키 순서와 같음).
    assert r1["staged"] == 245 + 429 + 412 + 935 + 40 == 2061


def test_week_saving_barely_moves_with_token_rule(skills):
    """토큰 규칙 세 가지에서 절감률이 모두 71.4%. 절대 토큰은 근사에 민감해도 비율은 안정적이다."""
    savings = [round(100 * float(B.week(skills, rule)["saving"]), 1) for rule in B.load_rules()]
    assert savings == [71.4, 71.4, 71.4]


# ---------- 대신 치르는 값 ----------

def test_round_trip_report_and_refund(skills):
    """읽을 때마다 호출이 늘어나는 비용을 손으로 맞춘 값과 대조한다. 캐싱을 쓰면 단계적 공개가 전부 싣기보다 비싸질 수 있는 사례(43,254 대 37,823)를 포함한다."""
    sz = B.sizes(skills)
    w = B.load_week()
    rt = w["round_trip"]
    rep = B.round_trip(sz, B.project_tokens(), w["conversations"][0], 24150, rt["tool_call_tokens"], rt["w"], rt["r"])
    assert rep["paste_all_inputs"] == [27934, 28024] and rep["paste_all_nocache"] == 55958
    assert rep["staged_inputs"] == [24824, 25286, 25976, 26321, 26411] and rep["staged_nocache"] == 128818
    assert rep["staged_cached"] == F("43254.45") and rep["paste_all_cached"] == F("37823.4")
    bat = B.round_trip(sz, B.project_tokens(), w["conversations"][0], 24150, rt["tool_call_tokens"], rt["w"], rt["r"],
                       batch=True)
    assert len(bat["staged_inputs"]) == 3 and bat["staged_nocache"] == 76521
    ref = B.round_trip(sz, B.project_tokens(), w["conversations"][1], 300, rt["tool_call_tokens"], rt["w"], rt["r"])
    assert ref["paste_all_nocache"] == 4084 and ref["staged_nocache"] == 3980
    assert ref["staged_cached"] == F("2357.6")


def test_cache_bill_hand():
    """_bill 손계산: 첫 호출 1,000은 쓰기 1.25배, 둘째 호출은 앞 1,000을 읽기(0.1배), 늘어난 100을 쓰기."""
    no, ca = B._bill([1000, 1100], F(5, 4), F(1, 10))
    assert no == 2100 and ca == F(5, 4) * 1000 + F(1, 10) * 1000 + F(5, 4) * 100


# ---------- 한 파일에 몰아 넣기 ----------

def test_one_file_bytes(skills):
    """모든 내용을 한 파일에 몰아 넣어도 12,083바이트로 Codex 한도 32 KiB(32,768) 미만. 한글은 3바이트."""
    text = B.all_in_one_file_text(skills)
    assert B.utf8_bytes(text) == 12083 and B.utf8_bytes(text) < 32768
    assert len("가".encode("utf-8")) == 3


# ---------- 기록 파일 ----------

def test_vendor_file_has_urls_and_date():
    """공급자 한도 값에 출처 URL과 확인일이 있는지, 핵심 수치가 문서값과 같은지 본다."""
    v = json.loads(B.VENDOR_FILE.read_text(encoding="utf-8"))
    assert v["checked"] == "2026-10-10" and B.VENDOR_FILE.name == "vendors_2026-10-10.json"
    for k, item in v.items():
        if isinstance(item, dict):
            assert item["checked"] == "2026-10-10"
            assert any(str(x).startswith("https://") for x in item.values())
    assert v["spec"]["name_max"] == 64 and v["spec"]["description_max"] == 1024
    assert v["codex"]["project_doc_max_bytes"] == 32 * 1024
    assert v["claude_code"]["listing_truncate_chars"] == 1536


def test_results_file_matches_modules(skills):
    """저장된 결과 파일의 합계가 모듈 재계산과 같은지 본다."""
    res = json.loads((HERE / "results" / "ch04.json").read_text(encoding="utf-8"))
    assert res["e4_week"]["paste_all_total"] == B.week(skills)["paste_all_total"]
    assert res["e4_week"]["staged_total"] == B.week(skills)["staged_total"]
    assert res["e3_sizes"]["sum_meta"] == sum(x["meta"] for x in B.sizes(skills))
    assert res["e8_crosscheck"]["mismatches"] == 0
