"""1인 개발 앱 출시 실전 5장 실험. `uv run python webapp/ch05_static_supabase/experiments.py` 로 실행한다.

E0 구성: 어느 파일이 어디에 올라가고, 어떤 키가 어디에 있나(브라우저 코드에 AI 키가 없는가)
E1 스토리지 정책: 같은 요청 13개를 정책 없음 / 정책 4개로 보내기
E2 덮어쓰기(upsert)는 INSERT 정책만으로 되나
E3 서명된 주소의 유효 시간
E4 대조: 파이썬 정책과 sqlite WHERE 절이 같은 파일을 고르나
E5 Edge Function 모형: 요청 12개의 응답 코드, 요금에 드는 호출 수, 결과 그림을 누가 받나
E6 무료 한도와 유료 전환 지점(Supabase·Vercel·Netlify), 가상 AI 비용
E7 벤더 종속: 규칙 중 Supabase 전용 함수를 쓰는 비율
E8 일시 정지 모형: 7일 연속 DB 요청이 없으면 위험(모형 가정)
E9 연습 문제 답과 본문의 중간 계산
E10 카드 표·사용량 표 정책: 누가 어느 카드를 읽고, 누가 쓸 수 있나

표준 라이브러리만 쓴다. 네트워크와 진짜 키 없이 돌고 결과는 매번 같다.
"""

import json
import platform
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import webapp_ch05_costs as C  # noqa: E402
import webapp_ch05_edge as E  # noqa: E402
import webapp_ch05_rules as R  # noqa: E402

OUT = HERE / "results" / "ch05.json"
f = C.to_float


def e0_layout():
    pub = HERE / "site" / "public"
    files = sorted(p.relative_to(pub).as_posix() for p in pub.rglob("*") if p.is_file())
    bundle = "".join((pub / x).read_text(encoding="utf-8") for x in files)
    fn = (HERE / "supabase" / "functions" / "generate-background" / "index.ts").read_text(encoding="utf-8")
    env = (HERE / "supabase" / "env.example").read_text(encoding="utf-8")
    return {
        "static_files": files,
        "static_bytes": sum((pub / x).stat().st_size for x in files),
        "bundle_has_publishable": "DEMO_SUPABASE_PUBLISHABLE_0005" in bundle,
        "bundle_has_ai_key": E.ENV["AI_IMAGE_KEY"] in bundle,
        "bundle_has_ai_key_name": "AI_IMAGE_KEY" in bundle,
        "function_reads_env": "Deno.env.get('AI_IMAGE_KEY')" in fn,
        "function_has_key_value": E.ENV["AI_IMAGE_KEY"] in fn,
        "env_example_has_key": E.ENV["AI_IMAGE_KEY"] in env,
    }


def e6_costs():
    pu = C.per_user()
    walls = C.free_walls()
    grid = []
    for mau in (100, 250, 600, 2000, 20000, 120000):
        row = {"mau": mau}
        for month in (1, 6, 12):
            used, over = C.free_status(mau, month)
            row[f"m{month}_over"] = over
        row["storage_full_month"] = C.first_month_over(walls["storage_user_months"], mau)
        row["db_full_month"] = C.first_month_over(walls["db_user_months"], mau)
        pc = C.pro_cost(mau, 12)
        row["pro_m12_total"] = f(pc["total"], 2)
        row["pro_m12_lines"] = {k: f(v, 2) for k, v in pc["lines"].items()}
        row["pro_m12_capped_blocked"] = C.pro_cost(mau, 12, spend_cap_off=False)["blocked_with_cap"]
        row["ai_cost_month"] = f(C.ai_cost(mau), 2)
        nf = C.netlify_free(mau)
        row["netlify_credits"] = f(nf["total"], 2)
        row["netlify_within"] = nf["within"]
        grid.append(row)
    nf0 = C.netlify_free(0)
    return {
        "per_user": {k: f(v) for k, v in pu.items()},
        "free_walls": {k: f(v, 2) for k, v in walls.items()},
        "grid": grid,
        "vercel_hobby": {k: f(v, 1) for k, v in C.vercel_hobby_mau().items()},
        "netlify": {"deploy_credits": f(nf0["parts"]["deploys"]), "per_user_credit": f(nf0["per_user_credit"]),
                    "max_mau_10_deploys": f(nf0["max_mau_at_deploys"], 1),
                    "max_mau_20_deploys": f(C.netlify_free(0, 20)["max_mau_at_deploys"], 1)},
        "commercial_month1": {"vercel_pro_seat": 20, "supabase_pro": 25, "total_min": 45},
    }


def e9_exercises():
    """연습 문제 답과 본문에 나오는 중간 계산."""
    pu = C.per_user()
    u = C.USAGE["per_active_user_month"]
    kb = C.USAGE["kb_per_mb"]
    upload_mb = u["uploads"] * u["upload_kb"] / kb
    ai_mb = u["ai_backgrounds"] * u["ai_png_kb"] / kb
    cap = C.PRICING["supabase"]["free"]["storage_gb"] * C.USAGE["mb_per_gb"]
    m = 100
    fixed = m * ai_mb                         # AI 그림은 최근 한 달 치만 남김
    months = (cap - fixed) / (m * upload_mb)  # 올린 그림만 쌓임
    ex1 = {"mau": 400, "egress_mb": f(400 * pu["egress_mb"]), "storage_m1_mb": f(400 * pu["storage_mb"]),
           "storage_m2_mb": f(400 * pu["storage_mb"] * 2), "storage_full_month": C.first_month_over(500, 400),
           "egress_over": "egress" in C.free_status(400, 1)[1]}
    ex2 = {"mau": m, "upload_mb": upload_mb, "ai_mb": ai_mb, "fixed_mb": fixed, "left_mb": cap - fixed,
           "per_month_mb": m * upload_mb, "months": months, "full_month": int(months) + 1}
    p120 = C.pro_cost(120000, 12)
    p20 = C.pro_cost(20000, 12)
    used120, _ = C.free_status(120000, 12)
    used20, _ = C.free_status(20000, 12)
    mid = {
        "m120_storage_gb": f(used120["storage_gb"]), "m120_storage_over": f(p120["over"]["storage_gb"]),
        "m120_egress_gb": f(used120["egress_gb"]), "m120_egress_over": f(p120["over"]["egress_gb"]),
        "m120_edge_calls": used120["edge_calls"], "m120_db_gb": f(used120["db_mb"] / 1000),
        "m20_storage_gb": f(used20["storage_gb"]), "m20_storage_over": f(p20["over"]["storage_gb"]),
        "m20_egress_gb": f(used20["egress_gb"]),
        "netlify_20_deploys_credits": f(C.netlify_free(0, 20)["parts"]["deploys"]),
        "free_egress_mb": C.PRICING["supabase"]["free"]["egress_gb"] * C.USAGE["mb_per_gb"],
        "free_storage_mb": C.PRICING["supabase"]["free"]["storage_gb"] * C.USAGE["mb_per_gb"],
    }
    o = C.PRICING["supabase"]["pro"]["overage"]
    docs = {  # 공식 문서의 계산 예제를 같은 단가로 다시 계산
        "mau_160k_overage_usd": f((160000 - 100000) * C.Fraction(str(o["mau_usd_each"])), 2),
        "edge_1000001_packs": -(-1_000_001 // 1_000_000),
        "storage_188gb_overage_usd": f(188 * C.Fraction(str(o["storage_usd_per_gb_month"])), 2),
    }
    return {"ex1": ex1, "ex2": ex2, "mid": mid, "docs_examples": docs}


def e7_lockin():
    text = R.POLICIES_SQL.read_text(encoding="utf-8")
    blocks = re.findall(r'create policy "[^"]+".*?;', text, flags=re.S)
    special = [b for b in blocks if re.search(r"\bauth\.(uid|jwt)\(|\bstorage\.(foldername|filename|extension)\(", b)]
    on_storage = [b for b in blocks if "storage.objects" in b]
    app = (HERE / "site" / "public" / "app.js").read_text(encoding="utf-8")
    fn = (HERE / "supabase" / "functions" / "generate-background" / "index.ts").read_text(encoding="utf-8")
    return {
        "policies": len(blocks),
        "use_supabase_functions": len(special),
        "on_storage_objects": len(on_storage),
        "bucket_inserts": len(re.findall(r"insert into storage\.buckets", text)),
        "app_supabase_calls": len(re.findall(r"supabase\.(from|storage|functions)", app)),
        "function_supabase_calls": len(re.findall(r"ctx\.supabaseAdmin|withSupabase\(", fn)),
        "function_deno_env": len(re.findall(r"Deno\.env\.get", fn)),
    }


def e8_pause():
    # 30일 동안 하루 DB 요청 수(가상). 모형 규칙: 최근 7일이 모두 0이면 "정지 위험"
    daily = {
        "매일 쓰는 편집기": [12] * 30,
        "주말 행사용 앱": ([40, 35, 0, 0, 0, 0, 0] + [0] * 5 + [50, 45] + [0] * 16),
        "개발용 프로젝트(휴가)": [5] * 10 + [0] * 20,
    }
    out = {}
    for name, xs in daily.items():
        flag = None
        run = 0
        for d, x in enumerate(xs, start=1):
            run = run + 1 if x == 0 else 0
            if run >= 7 and flag is None:
                flag = d
        out[name] = {"days": len(xs), "requests": sum(xs), "first_risk_day": flag}
    return out


def main():
    e5 = E.run_scenarios()
    e5["without_verify_jwt"] = E.run_without_verify_jwt()
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "checked": "2026-10-10"},
        "e0_layout": e0_layout(),
        "e1_storage": R.run_storage_scenarios(),
        "e1_blocked_with_policies": sum(1 for x in R.run_storage_scenarios()
                                        if x["policies"] == "정책 4개" and x["result"] != "허용"),
        "e2_upsert": R.run_upsert_needs_select_update(),
        "e3_signed_url": R.run_signed_url(),
        "e4_crosscheck": R.crosscheck_sqlite(),
        "e5_edge": e5,
        "e6_costs": e6_costs(),
        "e7_lockin": e7_lockin(),
        "e8_pause": e8_pause(),
        "e9_exercises": e9_exercises(),
        "e10_tables": R.run_table_scenarios(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
