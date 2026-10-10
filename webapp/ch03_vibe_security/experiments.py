"""1인 개발 앱 출시 실전 3장 실험. `uv run python webapp/ch03_vibe_security/experiments.py` 로 실행한다.

E0 점검 목록 다섯 가지: 고치기 전/후 통과 수
E1 RLS: 끔 / 켜기만 함 / 정책까지, 로그인 안 함·alice·bob이 읽고 쓰는 결과
E2 AI 키: 브라우저가 받는 번들에 비밀 키가 몇 번 나오나
E3 관리자 주소: 로그인 안 함·일반 사용자·관리자가 관리 화면과 관리용 API에 닿는가
E4 업로드: 시험 파일 11개를 고치기 전/후 서버에 올린 결과
E5 사용량 제한: 토큰 버킷(B=5, 12초에 1개) 손계산·시뮬레이션·HTTP 대조, 1장식 전체 횟수 세기와 비교
E6 대조: 파이썬 RLS와 sqlite WHERE 절이 같은 행을 고르나, 허용 수가 B + rT를 넘지 않나

표준 라이브러리만 쓴다. 서버는 127.0.0.1의 빈 포트에 잠깐 띄웠다 끈다. 시계는 가짜 시계라 결과가 매번 같다.
"""

import json
import math
import platform
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import webapp_ch03_app as A  # noqa: E402
import webapp_ch03_ratelimit as R  # noqa: E402
import webapp_ch03_rls as L  # noqa: E402
import webapp_ch03_upload as U  # noqa: E402

OUT = HERE / "results" / "ch03.json"
TOKENS = {"로그인 안 함": None, "alice": "demo-session-alice", "bob": "demo-session-bob",
          "관리자": "demo-session-admin"}
UNIT_PRICE_WON = 50  # AI 배경 한 장의 가상 단가(실제 가격이 아니에요)


def with_fixes(**kw):
    f = dict(A.BEFORE)
    f.update(kw)
    return f


# ---------- E1 RLS ----------

def e1_rls():
    out = {}
    for mode in ("off", "on_no_policy", "on_policies"):
        rec = {"read": {}, "writes": {}}
        with A.running(with_fixes(rls=mode)) as (base, st):
            for who in ("로그인 안 함", "alice", "bob"):
                code, _, body = A.request(base, "/rest/v1/cards", token=TOKENS[who], apikey=A.PUBLISHABLE_KEY)
                rows = json.loads(body)
                rec["read"][who] = {"status": code, "rows": len(rows), "ids": [r["id"] for r in rows]}
            code, _, _ = A.request(base, "/rest/v1/cards", apikey=None)
            rec["no_api_key_status"] = code
        # 쓰기 시험은 하나씩 새 서버에서(앞 시험이 뒤 시험에 영향을 주지 않게)
        tests = [
            ("alice가 bob 이름으로 카드 만들기", "POST", "/rest/v1/cards", "alice",
             {"owner_id": "u-bob", "title": "남의 이름으로 만든 카드"}),
            ("alice가 자기 이름으로 카드 만들기", "POST", "/rest/v1/cards", "alice",
             {"owner_id": "u-alice", "title": "새 카드"}),
            ("로그인 안 하고 카드 만들기", "POST", "/rest/v1/cards", "로그인 안 함",
             {"owner_id": "u-alice", "title": "익명 카드"}),
            ("alice가 bob의 카드 4번 제목 고치기", "PATCH", "/rest/v1/cards?id=4", "alice", {"title": "바뀐 제목"}),
            ("alice가 자기 카드 2번 주인을 bob으로 바꾸기", "PATCH", "/rest/v1/cards?id=2", "alice",
             {"owner_id": "u-bob"}),
            ("alice가 자기 카드 2번 제목 고치기", "PATCH", "/rest/v1/cards?id=2", "alice", {"title": "고친 초안"}),
            ("alice가 bob의 카드 5번 지우기", "DELETE", "/rest/v1/cards?id=5", "alice", None),
        ]
        for name, method, path, who, body in tests:
            with A.running(with_fixes(rls=mode)) as (base, st):
                code, _, resp = A.request(base, path, method=method, body=body, token=TOKENS[who],
                                          apikey=A.PUBLISHABLE_KEY)
                after = st.cards.all_rows()
            changed = after != [dict(zip(("id", "owner_id", "title", "is_public"), r)) for r in L.SEED_CARDS]
            rec["writes"][name] = {"status": code, "response": json.loads(resp), "table_changed": changed}
        out[mode] = rec
    # 고치기 전 번들은 화면에서만 거른다: 받은 행과 화면에 그린 행
    rows_off = out["off"]["read"]["alice"]["rows"]
    alice_own = sum(1 for c in L.SEED_CARDS if c[1] == "u-alice")
    out["client_filter_before"] = {"rows_in_response": rows_off, "rows_on_screen": alice_own,
                                   "other_users_rows_in_response": rows_off - alice_own}
    return out


# ---------- E2 AI 키 ----------

def e2_keys():
    out = {}
    for label, fixes in (("before", A.BEFORE), ("after", A.AFTER)):
        with A.running(fixes) as (base, st):
            blob = b"".join(A.request(base, p)[2] for p in ("/", "/app.js"))
        out[label] = {
            "downloaded_bytes": len(blob),
            "ai_secret_key_count": blob.count(A.AI_SECRET_KEY.encode()),
            "publishable_key_count": blob.count(A.PUBLISHABLE_KEY.encode()),
            "admin_path_count": blob.count(b'"/admin"'),
            "calls_ai_provider_directly": blob.count(b"ai.example.invalid") > 0,
        }
    return out


# ---------- E3 관리자 ----------

ADMIN_ENDPOINTS = [("GET", "/admin"), ("GET", "/admin/api/users"), ("POST", "/admin/api/cards/delete?id=5")]


def e3_admin():
    out = {}
    for label, fixes in (("before", A.BEFORE), ("after", A.AFTER)):
        table = {}
        for who in ("로그인 안 함", "alice", "관리자"):
            row = {}
            for method, path in ADMIN_ENDPOINTS:
                with A.running(fixes) as (base, st):
                    code, _, body = A.request(base, path, method=method, token=TOKENS[who])
                row[f"{method} {path}"] = code
            table[who] = row
        out[label] = table
    # 브라우저가 보낸 역할 표시는 보지 않는다(고친 뒤)
    with A.running(A.AFTER) as (base, st):
        code, _, _ = A.request(base, "/admin/api/users", token=TOKENS["alice"],
                               headers={"X-User-Role": "admin"})
    out["after_ignores_client_role_header"] = code
    with A.running(A.BEFORE) as (base, st):
        code, _, body = A.request(base, "/admin/api/users")
    out["before_users_listed_to_anon"] = len(json.loads(body))
    return out


# ---------- E4 업로드 ----------

def e4_upload():
    rows = []
    for label, name, ctype, data in U.samples():
        rec = {"sample": label, "filename": name, "content_type": ctype, "bytes": len(data)}
        for mode, fixes in (("before", A.BEFORE), ("after", A.AFTER)):
            with A.running(fixes) as (base, st):
                from urllib.parse import quote
                code, _, body = A.request(base, "/api/upload?name=" + quote(name), method="POST", body=data,
                                          token=TOKENS["alice"], headers={"Content-Type": ctype})
                resp = json.loads(body)
                served = None
                if resp.get("ok"):
                    c2, h2, b2 = A.request(base, resp["url"])
                    served = {"status": c2, "content_type": h2.get("Content-Type"),
                              "nosniff": h2.get("X-Content-Type-Options") == "nosniff", "same_bytes": b2 == data}
            rec[mode] = {"status": code, "accepted": bool(resp.get("ok")), "reasons": resp.get("reasons", []),
                         "url": resp.get("url"), "served": served}
        rec["validate"] = U.validate(name, ctype, data)
        rows.append(rec)
    summary = {
        "samples": len(rows),
        "accepted_before": sum(r["before"]["accepted"] for r in rows),
        "accepted_after": sum(r["after"]["accepted"] for r in rows),
        "served_as_html_before": sum(1 for r in rows if r["before"]["served"]
                                     and r["before"]["served"]["content_type"].startswith("text/html")),
        "served_as_svg_before": sum(1 for r in rows if r["before"]["served"]
                                    and r["before"]["served"]["content_type"].startswith("image/svg")),
        "max_bytes": U.MAX_BYTES,
    }
    return {"rows": rows, "summary": summary}


# ---------- E5 사용량 제한 ----------

def frac(x):
    return float(x) if x.denominator != 1 else int(x)


def e5_rate():
    s1_times = [R.F(k, 2) for k in range(30)]          # 0.5초 간격 30번(0초 ~ 14.5초)
    s2_times = [R.F(k) for k in range(3600)]           # 1초 간격 3,600번(1시간)
    out = {"capacity_B": R.CAPACITY, "refill_per_sec_r": str(R.REFILL_PER_SEC),
           "refill_seconds_per_token": int(1 / R.REFILL_PER_SEC), "unit_price_won_virtual": UNIT_PRICE_WON}
    for key, times in (("s1_burst_30_in_15s", s1_times), ("s2_1_per_sec_for_1h", s2_times)):
        T = times[-1] - times[0]
        sim = R.simulate(R.bucket_take(), times)
        none = R.simulate(lambda u, t: (True, 0), times)
        out[key] = {
            "requests": len(times), "span_seconds": frac(T),
            "allowed_no_limit": none["allowed"], "allowed_bucket": sim["allowed"],
            "rejected_bucket": sim["rejected"], "first_retry_after": sim["first_retry_after"],
            "bound_B_plus_rT": str(R.upper_bound(T)), "bound_floor": math.floor(R.upper_bound(T)),
            "allowed_at_first_10": [frac(t) for t in sim["allowed_at"][:10]],
            "cost_won_no_limit": none["allowed"] * UNIT_PRICE_WON,
            "cost_won_bucket": sim["allowed"] * UNIT_PRICE_WON,
        }
    # 1장 서버처럼 전체 횟수만 세면(한도 20): alice가 20번 쓰고 나면 bob도 막힌다
    g = R.GlobalCounter(20)
    alice_g = sum(g.take("alice")[0] for _ in range(20))
    bob_g = g.take("bob")[0]
    clock = R.FakeClock(0)
    p = R.PerUserLimiter(clock)
    alice_p = 0
    for k in range(20):
        clock.set(R.F(k, 2))
        alice_p += p.take("alice")[0]
    clock.set(10)
    bob_p = p.take("bob")[0]
    out["s3_two_users"] = {"global_counter": {"alice_allowed_of_20": alice_g, "bob_first_allowed": bob_g},
                           "per_user_bucket": {"alice_allowed_of_20": alice_p, "bob_first_allowed": bob_p}}
    # HTTP로 같은 S1을 다시: 고치기 전/후 서버
    http = {}
    for label, fixes in (("before", A.BEFORE), ("after", A.AFTER)):
        clock = R.FakeClock(0)
        codes, retry = [], None
        with A.running(fixes, clock) as (base, st):
            for t in s1_times:
                clock.set(t)
                code, h, _ = A.request(base, "/api/generate-image", method="POST", body={"prompt": "가을 하늘"},
                                       token=TOKENS["alice"])
                codes.append(code)
                if code == 429 and retry is None:
                    retry = int(h.get("Retry-After"))
            generated = st.generated
            code_anon, _, _ = A.request(base, "/api/generate-image", method="POST", body={"prompt": "x"})
        http[label] = {"ok_200": codes.count(200), "too_many_429": codes.count(429),
                       "first_retry_after_header": retry, "images_generated": generated,
                       "anon_status": code_anon}
    out["http_s1"] = http
    return out


# ---------- E6 대조 ----------

def e6_crosscheck():
    mism = 0
    checked = 0
    for mode in ("off", "on_no_policy", "on_policies"):
        t = L.CardsTable(mode)
        for role, uid in (("anon", None), ("authenticated", "u-alice"), ("authenticated", "u-bob"),
                          ("authenticated", "u-nobody")):
            checked += 1
            if t.select(role, uid) != t.select_sql(role, uid):
                mism += 1
    over = 0
    grid = 0
    for gap_num in (1, 2, 3, 5, 7, 12, 13, 30):        # 요청 간격 0.25초 ~ 7.5초
        gap = R.F(gap_num, 4)
        for n in (1, 10, 50, 200):
            times = [gap * k for k in range(n)]
            sim = R.simulate(R.bucket_take(), times)
            grid += 1
            if sim["allowed"] > math.floor(R.upper_bound(times[-1] - times[0])):
                over += 1
    sql_names = L.policies_in_sql_file()
    py_names = [(p["name"], p["cmd"]) for p in L.POLICIES]
    return {"rls_python_vs_sqlite": {"checked": checked, "mismatches": mism},
            "bucket_bound": {"scenarios": grid, "over_bound": over},
            "policies_sql_matches_python": sql_names == py_names, "policies": len(py_names)}


def e0_checklist(e1, e2, e3, e4, e5):
    items = [
        ("RLS: 로그인 안 한 사람이 비공개 카드를 읽을 수 없다",
         e1["off"]["read"]["로그인 안 함"]["rows"] == 1, e1["on_policies"]["read"]["로그인 안 함"]["rows"] == 1),
        ("AI 비밀 키가 브라우저 번들에 없다",
         e2["before"]["ai_secret_key_count"] == 0, e2["after"]["ai_secret_key_count"] == 0),
        ("일반 사용자가 관리용 API에 닿지 않는다",
         e3["before"]["alice"]["GET /admin/api/users"] == 403, e3["after"]["alice"]["GET /admin/api/users"] == 403),
        ("그림이 아닌 파일은 업로드되지 않는다",
         e4["summary"]["accepted_before"] == 3, e4["summary"]["accepted_after"] == 3),
        ("한 사용자가 AI 배경을 몰아서 만들 수 없다(15초에 30번 시도)",
         e5["http_s1"]["before"]["too_many_429"] > 0, e5["http_s1"]["after"]["too_many_429"] > 0),
    ]
    return {"items": [{"check": n, "before_pass": b, "after_pass": a} for n, b, a in items],
            "before_pass": sum(b for _, b, _ in items), "after_pass": sum(a for _, _, a in items)}


def main():
    e1 = e1_rls()
    e2 = e2_keys()
    e3 = e3_admin()
    e4 = e4_upload()
    e5 = e5_rate()
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "checked": "2026-10-10", "server": "http.server (ThreadingHTTPServer, 127.0.0.1), sqlite3 :memory:"},
        "e0_checklist": e0_checklist(e1, e2, e3, e4, e5),
        "e1_rls": e1,
        "e2_keys": e2,
        "e3_admin": e3,
        "e4_upload": e4,
        "e5_rate_limit": e5,
        "e6_crosscheck": e6_crosscheck(),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res["e0_checklist"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
