"""1인 개발 앱 출시 실전 5장 검증. `uv run pytest -q webapp/ch05_static_supabase` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 네트워크와 진짜 키 없이 돈다.
"""

import json
import math
import re
from fractions import Fraction
from pathlib import Path

import pytest

import webapp_ch05_costs as C
import webapp_ch05_edge as E
import webapp_ch05_rules as R

HERE = Path(__file__).parent
# experiments.py가 저장한 결과 파일. 아래 테스트 일부는 지금 코드 결과와 같은지(최신인지)도 확인한다.
RESULTS = json.loads((HERE / "results" / "ch05.json").read_text(encoding="utf-8"))
PRICING = C.PRICING


# ---------- 실습 폴더에 진짜 모양의 키가 없다 ----------

# 진짜 키 접두사(일부). 실습 폴더 전체를 훑어 접두사 + 8글자 이상, 또는 JWT 모양이 없는지 본다.
REAL_PREFIXES = ["sb_secret_", "sb_publishable_", "sk_live_", "sk_test_", "pk_live_", "rk_live_"]


def test_no_real_key_shapes_on_disk():
    pat = re.compile(r"(?<![A-Za-z0-9_\-])(" + "|".join(map(re.escape, REAL_PREFIXES)) + r")[A-Za-z0-9_\-]{8,}")
    jwt = re.compile(r"eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.")
    for p in HERE.rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            text = p.read_text(encoding="utf-8", errors="replace")
            assert not pat.search(text), p
            assert not jwt.search(text), p


def test_secrets_are_demo_values():
    assert E.ENV["AI_IMAGE_KEY"].startswith("DEMO_")
    assert E.JWT_SECRET.startswith(b"DEMO_")


# ---------- 정책 파일과 파이썬 정책이 같은 뜻 ----------

def test_sql_policy_names_match_python():
    sql = R.policies_in_sql_file()
    storage = [(n, cmd) for n, tbl, cmd in sql if tbl == "storage.objects"]
    assert storage == [(p["name"], p["cmd"]) for p in R.STORAGE_POLICIES]
    others = [(tbl, cmd) for n, tbl, cmd in sql if tbl != "storage.objects"]
    # 카드: 읽기 1(공개 or 내 것), 만들기 1 / 사용량: 읽기 1, 쓰기 정책 없음
    # 표 정책 3개: 카드 읽기, 카드 만들기, 사용량 읽기. 사용량에 쓰기 정책은 의도적으로 없다. 스토리지 4개와 합쳐 7개.
    assert others == [("public.cards", "select"), ("public.cards", "insert"), ("public.ai_usage", "select")]
    assert len(sql) == 7
    text = R.POLICIES_SQL.read_text(encoding="utf-8")
    assert "using ( is_public = true or (select auth.uid()) = owner_id )" in text


# ---------- 카드 표·사용량 표 ----------

def test_card_reads_and_writes():
    t = R.run_table_scenarios()
    # 시작 카드: 공개는 1번(alice), 4번(bob). alice는 공개 2장 + 자기 비공개 2장 = [1, 2, 3, 4].
    assert t["reads"]["로그인 안 함"]["cards"] == [1, 4]          # 공개 카드만
    assert t["reads"]["alice"]["cards"] == [1, 2, 3, 4]           # 공개 + 내 카드
    assert t["reads"]["bob"]["cards"] == [1, 4, 5]
    assert t["reads"]["비밀 키"]["cards"] == [1, 2, 3, 4, 5]
    assert t["writes"]["alice가 bob 이름으로 카드 만들기"] == "거부"
    assert t["writes"]["로그인 안 한 사람이 카드 만들기"] == "거부"
    assert t["sql_agree"] is True
    assert RESULTS["e10_tables"] == t


def test_ai_usage_has_no_write_policy():
    t = R.run_table_scenarios()
    # 사용량 표는 쓰기 정책이 없어 default deny. 비밀 키(서버 함수)만 고친다.
    assert t["writes"]["alice가 자기 사용량 늘리기"] == "거부"   # default deny
    assert t["writes"]["비밀 키로 사용량 늘리기"] == "허용"
    assert t["reads"]["alice"]["usage_rows"] == 1 and t["reads"]["로그인 안 함"]["usage_rows"] == 0
    text = R.POLICIES_SQL.read_text(encoding="utf-8")
    assert not re.search(r"on public\.ai_usage for (insert|update|delete|all)", text)


def test_exercise_answers():
    ex = RESULTS["e9_exercises"]
    assert ex["ex1"]["storage_full_month"] == 2 and ex["ex1"]["egress_over"] is False
    # 연습 2: MAU 100, AI 그림 100 x 1.2MB = 120MB는 최근 한 달 치만 남는 고정분, 남은 880MB를 매달 80MB(100 x 0.8)씩 채워 11달 끝에 딱 1GB, 12번째 달에 넘친다.
    assert ex["ex2"]["fixed_mb"] == 120 and ex["ex2"]["left_mb"] == 880 and ex["ex2"]["full_month"] == 12
    # MAU 12만, 12달 뒤: 스토리지 2,880GB에서 Pro 포함 100GB를 뺀 2,780GB, egress 1,200GB에서 250GB를 뺀 950GB 초과.
    assert ex["mid"]["m120_storage_over"] == 2780 and ex["mid"]["m120_egress_over"] == 950
    assert RESULTS["e1_blocked_with_policies"] == 8


def test_helper_functions_match_docs_example():
    # 문서 예시: public/subfolder/avatar.png
    # Supabase 문서 예시와 같은 결과인지 확인(폴더 목록은 파일 이름 제외).
    assert R.foldername("public/subfolder/avatar.png") == ["public", "subfolder"]
    assert R.filename("public/subfolder/avatar.png") == "avatar.png"
    assert R.extension("public/subfolder/avatar.png") == "png"


# ---------- 스토리지 정책 ----------

def _results(label):
    return {r["case"]: r["result"] for r in RESULTS["e1_storage"] if r["policies"] == label}


def test_no_policy_denies_uploads_but_public_bucket_reads():
    r = _results("정책 없음")
    # 정책이 없으면 올리기는 모두 거부. 공개 버킷의 읽기만 정책 없이 된다. 비밀 키는 정책을 건너뛴다.
    assert all(v.startswith("거부") for k, v in r.items() if "올리기" in k)
    assert r["키 없이 템플릿 공개 주소로 받기"] == "허용"
    assert r["비밀 키로 bob 파일 받기"] == "허용"


def test_policies_allow_only_own_folder_images():
    r = _results("정책 4개")
    assert r["alice가 내 폴더에 png 올리기"] == "허용"
    assert r["alice가 내 폴더에 txt 올리기"].startswith("거부")
    assert r["alice가 bob 폴더에 올리기"].startswith("거부")
    assert r["로그인 안 한 사람이 올리기"].startswith("거부")
    assert r["alice가 템플릿 버킷에 올리기"].startswith("거부")
    assert r["bob이 alice 파일 받기"].startswith("거부")
    assert r["bob이 alice 파일 지우기"] == "0개(보이지 않음)"
    assert r["키 없이 비공개 버킷 주소로 받기"].startswith("거부")
    # 정책 4개 상황에서 허용되는 요청 5개(내 png 올리기, 내 파일 받기, 공개 템플릿 받기, 내 png 덮어쓰기, 비밀 키 받기).
    assert sum(v == "허용" for v in r.values()) == 5


def test_upsert_needs_select_and_update():
    assert RESULTS["e2_upsert"] == {"insert만": "거부(upsert)", "insert+select+update": "덮어씀"}


def test_signed_url_expires():
    assert RESULTS["e3_signed_url"] == {"0": "받음", "599": "받음", "600": "만료", "bob_create": "거부"}


def test_python_and_sqlite_agree():
    c = R.crosscheck_sqlite()
    # 시험 경로 x 3사용자 x (select, insert) = 비교 6번 모두 일치해야 한다.
    assert c["agree"] == c["comparisons"] == 6
    assert c["by_user"]["anon"] == {"select": 0, "insert_ok": 0}


def test_results_are_fresh():
    assert RESULTS["e1_storage"] == R.run_storage_scenarios()
    assert RESULTS["e4_crosscheck"] == R.crosscheck_sqlite()


# ---------- Edge Function 모형 ----------

def test_edge_scenarios():
    e = E.run_scenarios()
    # 기대 코드: OPTIONS 204, 토큰 없음·위조·만료 401(플랫폼), 201자 400, alice 5번 200, 6번째 429, bob 200.
    assert [s["status"] for s in e["steps"]] == [204, 401, 401, 401, 400, 200, 200, 200, 200, 200, 429, 200]
    # 요금에 세는 호출 8 = 12 - OPTIONS 1 - 플랫폼이 막은 3.
    assert e["billed_invocations"] == 8 and e["rejected_by_platform"] == 3 and e["preflight"] == 1
    assert e["alice_usage"] == E.DAILY_LIMIT == 5
    assert e["who_can_read_alice_ai"] == {"alice": "받음", "bob": "거부"}
    assert e["key_in_any_response"] is False
    for k in ("steps", "billed_invocations", "ai_files"):
        assert RESULTS["e5_edge"][k] == e[k]


def test_jwt_model():
    t = E.make_token("u-alice", 100)
    assert E.verify_token(t, 99)["sub"] == "u-alice"
    # exp = 100, now = 100은 만료(경계: exp <= now). 99는 유효.
    assert E.verify_token(t, 100) is None  # 만료
    head, body, sig = t.split(".")
    # 서명 끝 두 글자를 바꾼 위조 토큰은 거부.
    assert E.verify_token(f"{head}.{body}.{sig[:-2]}AA", 50) is None
    assert E.verify_token("not-a-token", 50) is None


def test_fake_ai_needs_key_and_makes_png():
    png = E.fake_ai_png("x", E.ENV["AI_IMAGE_KEY"])
    # PNG 서명으로 시작하고 IEND 청크(길이 0 + 종류 + CRC)로 끝나는지 확인.
    assert png.startswith(b"\x89PNG\r\n\x1a\n") and png.endswith(b"IEND\xaeB`\x82")
    with pytest.raises(PermissionError):
        E.fake_ai_png("x", "DEMO_WRONG")


def test_browser_bundle_has_no_ai_key():
    e0 = RESULTS["e0_layout"]
    assert e0["bundle_has_publishable"] and not e0["bundle_has_ai_key"] and not e0["bundle_has_ai_key_name"]
    assert e0["function_reads_env"] and not e0["function_has_key_value"]


def test_without_verify_jwt_still_billed():
    # verify_jwt를 끄면 토큰 없는 호출 3번도 함수 코드에 닿아 3번 모두 요금에 센다(응답은 401).
    assert E.run_without_verify_jwt() == {"billed_invocations": 3, "statuses": [401, 401, 401]}


# ---------- 가격표와 비용 계산 ----------

def test_pricing_file_has_sources_and_date():
    assert PRICING["checked"] == "2026-10-10"
    for vendor in ("supabase", "vercel", "netlify"):
        for url in PRICING[vendor]["sources"].values():
            assert url.startswith("https://")
    s = PRICING["supabase"]
    assert (s["free"]["db_size_mb"], s["free"]["storage_gb"], s["free"]["egress_gb"], s["free"]["mau"],
            s["free"]["edge_invocations"]) == (500, 1, 5, 50000, 500000)
    assert (s["pro"]["price_usd"], s["pro"]["storage_gb"], s["pro"]["egress_gb"], s["pro"]["mau"]) == (25, 100, 250, 100000)


def test_hand_calculations():
    pu = C.per_user()
    # 손계산: 스토리지 2x400 + 4x300 = 2000KB = 2MB, egress 30x300KB + 1MB = 10MB, DB 6 x 2KB = 12KB.
    assert pu["storage_mb"] == 2 and pu["egress_mb"] == 10 and pu["db_mb"] == Fraction(12, 1000)
    w = C.free_walls()
    # 무료 egress 5GB = 5000MB / 10MB = 500명. 스토리지는 1000MB / 2MB = 500 사용자-달. Edge는 50만 / 5 = 10만명.
    assert w["egress_mau"] == 500 and w["storage_user_months"] == 500 and w["edge_mau"] == 100000
    # 100명 x 2MB = 매달 200MB씩 쌓여 5달 끝에 딱 1GB(= 한도, 안 넘음), 6번째 달에 넘는다.
    assert C.first_month_over(500, 100) == 6       # 5달 끝에 딱 1GB, 6번째 달에 넘침
    assert C.first_month_over(500, 250) == 3
    # egress 500 x 10MB = 5GB로 한도와 같다: 넘친 것이 아니다(경계).
    assert C.free_status(500, 1)[1] == []          # 딱 5GB는 넘친 게 아님
    assert "egress" in C.free_status(501, 1)[1]


def test_docs_examples_reproduce():
    # MAU 문서 예: 160,000 MAU -> 60,000 x 0.00325 = 195달러
    # 문서의 MAU 초과 예시(16만 MAU = 10만 초과 6만 x 단가)가 같은 단가로 재현되는지 확인.
    assert 60000 * Fraction("0.00325") == 195
    # Edge 문서 예: 1,000,001회 넘으면 2묶음
    assert math.ceil(1_000_001 / 1_000_000) == 2
    # 스토리지 문서 예: 188GB 초과 x 0.0213 ≈ 4달러
    # 문서 예시: 188GB x 0.0213달러 = 4.0044 -> 약 4달러(float이라 반올림해서 비교).
    assert round(188 * 0.0213) == 4


def test_pro_cost_lines():
    c = C.pro_cost(120000, 12)
    # MAU 12만: (12만 - 10만) x 0.00325 = 65달러.
    assert float(c["lines"]["mau"]) == 65.0
    assert round(float(c["lines"]["storage"]), 2) == 59.21     # (2,880 - 100)GB x 0.0213
    assert float(c["lines"]["egress"]) == 85.5                  # (1,200 - 250)GB x 0.09
    assert c["lines"]["edge"] == 0                              # 600,000회 < 2,000,000
    # 합계 = 기본 25 + MAU 65 + 스토리지 59.21 + egress 85.5 + 0(edge) + DB 초과분. 반올림한 값 비교.
    assert round(float(c["total"]), 2) == 235.87
    assert C.pro_cost(600, 12)["total"] == 25
    # 스펜드 캡을 켠 상태에서는 초과분이 청구되지 않아 기본 25달러에서 멈춘다(대신 사용이 막힘).
    assert C.pro_cost(20000, 12, spend_cap_off=False) ["total"] == 25


def test_netlify_and_vercel():
    n = C.netlify_free(0)
    # Netlify: 월 배포 10번 x 배포당 15크레딧(가격표 값) = 150. 사용자 1명당 크레딧 0.052.
    assert n["parts"]["deploys"] == 150 and n["per_user_credit"] == Fraction(52, 1000)
    assert C.netlify_free(0, 20)["max_mau_at_deploys"] == 0
    v = C.vercel_hobby_mau()
    # Vercel Hobby: 전송 100GB / 사용자당 1MB = 10만명, 요청 100만 / 160 = 6,250명 중 요청 쪽이 먼저 한계.
    assert v["transfer_mau"] == 100000 and v["requests_mau"] == 6250


def test_results_cost_fresh():
    g = {r["mau"]: r for r in RESULTS["e6_costs"]["grid"]}
    assert g[100]["storage_full_month"] == 6 and g[100]["m1_over"] == []
    assert g[600]["m1_over"] == ["egress", "스토리지"]
    assert g[120000]["pro_m12_total"] == 235.87
    # (300 - 150) / 0.052 = 약 2884.6명. 소수 첫째 자리 반올림 값.
    assert RESULTS["e6_costs"]["netlify"]["max_mau_10_deploys"] == 2884.6


def test_lockin_and_pause():
    lk = RESULTS["e7_lockin"]
    assert lk["policies"] == 7 and lk["use_supabase_functions"] == 7 and lk["on_storage_objects"] == 4
    p = RESULTS["e8_pause"]
    assert p["매일 쓰는 편집기"]["first_risk_day"] is None
    # 모형 규칙: 요청이 0인 날이 7일 연속이면 위험. 3~9일이 0이므로 9일째가 첫 위험일(가정이며 실제 정지 규칙은 문서 확인 필요).
    assert p["주말 행사용 앱"]["first_risk_day"] == 9
