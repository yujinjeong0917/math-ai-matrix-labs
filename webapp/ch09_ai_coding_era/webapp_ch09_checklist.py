"""1인 개발 앱 출시 실전 9장: 1~8장을 묶은 "출시 전 점검표".

점검 항목마다 확인하는 방법이 셋 중 하나예요.

- 다시 검사 : 앞 장의 파일을 이 자리에서 다시 읽어 판정해요(앞 장 모듈의 순수 함수만 부르고, 서버는 띄우지 않아요).
- 결과 인용 : 앞 장 실습이 낸 results/chNN.json 값을 읽어 와 판정해요(JSON 위치를 함께 적어요).
- 사람 확인 : 파일만 보고는 답할 수 없는 항목이에요. 참고값만 붙이고 판정은 비워 둬요(checks/ch09.csv에 적어요).

두 판에 같은 점검표를 돌려요.

- 첫 판   : AI로 처음 빨리 만든 편집기. 1장 편집기(지도 파일 포함), 2장 dist_mistake, 3장 before,
            6장 처음 쓴 manifest, 7장 cardnews_mistake 를 모은 것이에요.
- 출시 후보: 고친 편집기. 2장 dist_good, 3장 after, 5장 정책·함수, 6장 PWA 공개 폴더, 7장 cardnews_good 이에요.

8장(첫 출시 플랫폼 고르기) 항목은 플랫폼·결제 방식을 고르는 판단이라 모두 사람 확인이에요.
data/ch08_manual_2026-10-10.json 에 공식 문서 값(2026-10-10 확인)을 두고, 8장 results/ch08.json 이 있으면
그 계산 값(4,900원 한 건에 남는 돈, 첫해 고정비, 선택 도구 결과)을 참고값으로 덧붙여요.
"""

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
WEBAPP = HERE.parent
CH = {
    1: WEBAPP / "ch01_everything_visible",
    2: WEBAPP / "ch02_api_keys",
    3: WEBAPP / "ch03_vibe_security",
    4: WEBAPP / "ch04_browser_vs_server",
    5: WEBAPP / "ch05_static_supabase",
    6: WEBAPP / "ch06_app_packaging",
    7: WEBAPP / "ch07_capacitor_apk",
}
CH08_MANUAL = HERE / "data" / "ch08_manual_2026-10-10.json"
CH08 = WEBAPP / "ch08_platform_choice"
CH08_RESULT = CH08 / "results" / "ch08.json"

for _d in (CH[2], CH[6], CH[7]):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import webapp_ch02_scan as K  # noqa: E402  (2장 키 점검기)
import webapp_ch06_manifest as M  # noqa: E402  (6장 manifest 검사기)
import webapp_ch07_config as C7  # noqa: E402  (7장 설정 점검기)
import webapp_ch07_signing as S7  # noqa: E402  (7장 서명 점검표)

# 1장 experiments.py 의 SERVER_ONLY_MARKERS 와 같은 글자예요(서버에만 둔 값).
SERVER_ONLY_MARKERS = {
    "가짜 AI 키": "DEMO_NOT_A_REAL_KEY_0000",
    "이미지 생성 지시문": "글자를 넣지 말 것",
    "원본 템플릿의 작업 메모": "작업 원본: 공지형 템플릿 v3",
    "하루 사용 한도 설정 이름": "daily_limit_per_user",
}

# 두 판이 가리키는 폴더
FIRST = {
    "public": CH[1] / "editor" / "public",
    "builds": [CH[2] / "project" / "dist_mistake", CH[1] / "editor" / "public", CH[3] / "app" / "public_before"],
    "manifest": CH[6] / "data" / "manifests" / "broken_first_draft.json",
    "android": CH[7] / "project" / "cardnews_mistake",
}
RELEASE = {
    "public": CH[6] / "pwa" / "public",
    "builds": [CH[2] / "project" / "dist_good", CH[5] / "site" / "public", CH[6] / "pwa" / "public"],
    "manifest": CH[6] / "pwa" / "public" / "manifest.json",
    "android": CH[7] / "project" / "cardnews_good",
}


def rel(p):
    return Path(p).relative_to(WEBAPP).as_posix()


def result_path(n):
    return CH[n] / "results" / f"ch{n:02d}.json"


def load_result(n):
    return json.loads(result_path(n).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get(d, path):
    """'e3_rls[0].rows' 같은 위치로 JSON 값을 꺼내요."""
    cur = d
    for part in path.replace("]", "").split("."):
        name, *idx = part.split("[")
        if name:
            cur = cur[name]
        for i in idx:
            cur = cur[int(i)]
    return cur


def ch08_refs():
    """8장 결과에서 참고값을 꺼내요. 파일이 없거나 모양이 바뀌었으면 None."""
    if not CH08_RESULT.is_file():
        return None
    try:
        r8 = json.loads(CH08_RESULT.read_text(encoding="utf-8"))
        pay = {row["id"]: row["net"] for row in r8["e1_one_payment"]}
        sel = r8["e5_select"]["cardnews_kr_subscription"]["platforms"]
        return {
            "c8_target_platform": {"candidate_now": {p["platform"]: p["candidate_now"] for p in sel},
                                   "where": "results/ch08.json: e5_select.cardnews_kr_subscription"},
            "c8_store_fee": {"gross_krw": r8["e1_one_payment"][0]["gross"],
                             "net_web_pg": pay.get("web_pg"), "net_gp_15_tier": pay.get("gp_15_tier"),
                             "where": "results/ch08.json: e1_one_payment[].net"},
            "c8_account_cost": {k: r8["e3_fixed"][k]["first_year_cash"] for k in ("web_only", "android", "ios_no_mac")},
        }
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None


# ---------- 다시 검사 ----------

def public_bytes(folder):
    return b"".join(p.read_bytes() for p in sorted(Path(folder).rglob("*")) if p.is_file())


def markers_in(folder):
    blob = public_bytes(folder)
    return {k: blob.count(v.encode()) for k, v in SERVER_ONLY_MARKERS.items()}


def source_map_traces(folder):
    folder = Path(folder)
    maps = sorted(rel(p) for p in folder.rglob("*.map"))
    lines = sorted(rel(p) for p in folder.rglob("*.js") if b"sourceMappingURL" in p.read_bytes())
    return {"map_files": maps, "sourcemap_lines": lines}


def html_comments(folder):
    out = {}
    for p in sorted(Path(folder).rglob("*.html")):
        n = p.read_text(encoding="utf-8").count("<!--")
        if n:
            out[rel(p)] = n
    return out


def key_scan(folders):
    rows = []
    for f in folders:
        s = K.summary(K.scan_dir(f, "demo"))
        rows.append({"folder": rel(f), "secret": s["secret"], "public": s["public"], "env_files": s["env_files"]})
    return rows


def manifest_check(path):
    text = Path(path).read_text(encoding="utf-8")
    public_dir = path.parent if path.name == "manifest.json" else None
    f = M.check_manifest(text, public_dir=public_dir)
    return {"installable": M.installable_in_chromium(f), "failed": M.failed(f)}


def android_check(folder):
    r = C7.check_project(folder)
    sc = S7.scorecard(folder)
    perms_flagged = sum(1 for x in r["findings"] if x["rule"] == "permission_unused")
    return {"blocking": r["blocking"], "review": r["review"], "permission_unused": perms_flagged,
            "signing_machine_pass": sc["machine_pass"], "signing_machine_total": sc["machine_total"],
            "signing_human_total": sc["human_total"]}


# ---------- 점검표 ----------

def item(ch, iid, gate, question, how, first, release, evidence, value=None, note=""):
    return {"ch": ch, "id": iid, "gate": gate, "question": question, "how": how,
            "first": first, "release": release, "evidence": evidence, "value": value, "note": note}


def build_items():
    r = {n: load_result(n) for n in CH}
    items = []

    # 1장 보이는 것
    mf, mr = markers_in(FIRST["public"]), markers_in(RELEASE["public"])
    items.append(item(1, "c1_server_only", "막음", "서버에만 둔 값(키·지시문·원본 템플릿 메모·한도 설정 이름)이 공개 폴더에 한 번도 없다",
                      "다시 검사", sum(mf.values()) == 0, sum(mr.values()) == 0,
                      f"{rel(FIRST['public'])} / {rel(RELEASE['public'])}", {"first": mf, "release": mr}))
    sf, sr = source_map_traces(FIRST["public"]), source_map_traces(RELEASE["public"])
    items.append(item(1, "c1_source_map", "막음", "공개 폴더에 지도 파일(.map)과 sourceMappingURL 줄이 없다",
                      "다시 검사", not (sf["map_files"] or sf["sourcemap_lines"]),
                      not (sr["map_files"] or sr["sourcemap_lines"]),
                      f"{rel(FIRST['public'])} / {rel(RELEASE['public'])}", {"first": sf, "release": sr}))
    hf, hr = html_comments(FIRST["public"]), html_comments(RELEASE["public"])
    items.append(item(1, "c1_html_comment", "확인", "배포할 HTML에 계획·가격 메모 같은 주석이 남아 있지 않다",
                      "다시 검사", not hf, not hr, f"{rel(FIRST['public'])} / {rel(RELEASE['public'])}",
                      {"first": hf, "release": hr}))

    # 2장 키
    kf, kr = key_scan(FIRST["builds"]), key_scan(RELEASE["builds"])
    items.append(item(2, "c2_secret_in_build", "막음", "빌드 결과물(브라우저로 내려가는 폴더)에 비밀 키 모양이 0개다",
                      "다시 검사", all(x["secret"] == 0 for x in kf), all(x["secret"] == 0 for x in kr),
                      "webapp_ch02_scan.scan_dir(demo)", {"first": kf, "release": kr}))
    items.append(item(2, "c2_env_in_build", "막음", "빌드 결과물에 .env 같은 설정 파일이 없다",
                      "다시 검사", all(x["env_files"] == 0 for x in kf), all(x["env_files"] == 0 for x in kr),
                      "webapp_ch02_scan.scan_dir(demo)",
                      {"first": [x["env_files"] for x in kf], "release": [x["env_files"] for x in kr]}))

    # 3장 다섯 가지 (결과 인용)
    ids3 = ["c3_rls", "c3_ai_key", "c3_admin", "c3_upload", "c3_rate_limit"]
    for k, (iid, row) in enumerate(zip(ids3, r[3]["e0_checklist"]["items"])):
        items.append(item(3, iid, "막음", row["check"], "결과 인용", row["before_pass"], row["after_pass"],
                          f"{rel(result_path(3))}: e0_checklist.items[{k}]",
                          {"before_pass": row["before_pass"], "after_pass": row["after_pass"]}))

    # 4장 어디서 돌릴지
    dec = {d["id"]: d for d in r[4]["e0_decisions"]}
    server = sorted(d["name"] for d in dec.values() if d["where"] == "서버")
    calls_direct = r[3]["e2_keys"]["before"]["calls_ai_provider_directly"]
    items.append(item(4, "c4_secret_on_server", "막음",
                      "비밀 키·원본·판단을 쓰는 기능(AI 배경, 최종 PNG, 보관 저장)이 서버에서 돈다",
                      "결과 인용", not calls_direct, not r[3]["e2_keys"]["after"]["calls_ai_provider_directly"],
                      f"{rel(result_path(4))}: e0_decisions / {rel(result_path(3))}: e2_keys.*.calls_ai_provider_directly",
                      {"server_features": server, "first_calls_ai_directly": calls_direct}))
    items.append(item(4, "c4_draft_vs_archive", "확인", "임시 저장은 브라우저, 기기를 바꿔도 남아야 하는 보관 저장은 서버에 둔다",
                      "결과 인용", None, dec["draft_save"]["where"] == "브라우저" and dec["archive_save"]["where"] == "서버",
                      f"{rel(result_path(4))}: e0_decisions(draft_save, archive_save)",
                      {"draft_save": dec["draft_save"]["where"], "archive_save": dec["archive_save"]["where"]}))
    vm = r[4]["e5_vercel_month"]
    items.append(item(4, "c4_month_cost", "확인", "내 사용량으로 한 달 서버비를 계산해 두었다(렌더링을 서버로 옮기면 얼마나 느는지 포함)",
                      "사람 확인", None, None, f"{rel(result_path(4))}: e5_vercel_month.*.bill_usd",
                      {"decided_bill_usd": vm["decided"]["bill_usd"], "server_preview_bill_usd": vm["server_preview"]["bill_usd"]},
                      "4장 가상 사용량(카드 20,000장) 기준 참고값"))

    # 5장 RLS·한도
    lay = r[5]["e0_layout"]
    items.append(item(5, "c5_key_in_function", "막음", "AI 키는 Edge Function 환경 변수에만 있고 정적 파일·함수 코드에는 값이 없다",
                      "결과 인용", None, (not lay["bundle_has_ai_key"]) and (not lay["function_has_key_value"]) and lay["function_reads_env"],
                      f"{rel(result_path(5))}: e0_layout", {k: lay[k] for k in ("bundle_has_ai_key", "function_has_key_value", "function_reads_env")}))
    w = r[5]["e10_tables"]["writes"]
    items.append(item(5, "c5_table_writes", "막음", "DB 쓰기 정책: 남의 이름으로 카드 만들기, 자기 사용량 늘리기가 거부된다",
                      "결과 인용", None, w["alice가 bob 이름으로 카드 만들기"] == "거부" and w["alice가 자기 사용량 늘리기"] == "거부",
                      f"{rel(result_path(5))}: e10_tables.writes", w))
    items.append(item(5, "c5_storage_policies", "막음", "파일 보관함에도 정책이 있어, 남의 폴더 올리기·로그인 없는 올리기 같은 경우가 막힌다",
                      "결과 인용", None, r[5]["e1_blocked_with_policies"] > 0,
                      f"{rel(result_path(5))}: e1_blocked_with_policies", {"blocked_with_policies": r[5]["e1_blocked_with_policies"]}))
    grid100 = [g for g in r[5]["e6_costs"]["grid"] if g["mau"] == 100][0]
    items.append(item(5, "c5_free_tier_month", "확인", "무료 한도가 넘치는 달과 상업적 사용에 드는 첫 달 비용을 내 숫자로 계산했다",
                      "사람 확인", None, None, f"{rel(result_path(5))}: e6_costs",
                      {"mau100_storage_full_month": grid100["storage_full_month"],
                       "commercial_month1_usd": r[5]["e6_costs"]["commercial_month1"]["total_min"]},
                      "5장 가상 사용 패턴 기준 참고값"))
    items.append(item(5, "c5_pause", "확인", "드문드문 쓰는 앱이라면 무료 프로젝트 일시 정지에 대비했다",
                      "사람 확인", None, None, f"{rel(result_path(5))}: e8_pause",
                      {"weekend_app_first_risk_day": r[5]["e8_pause"]["주말 행사용 앱"]["first_risk_day"]},
                      "5장 모형 기준 참고값"))

    # 6장 포장
    m6f, m6r = manifest_check(FIRST["manifest"]), manifest_check(RELEASE["manifest"])
    items.append(item(6, "c6_manifest", "막음", "manifest가 크롬 계열의 설치 조건을 모두 채운다",
                      "다시 검사", m6f["installable"], m6r["installable"],
                      f"{rel(FIRST['manifest'])} / {rel(RELEASE['manifest'])}", {"first": m6f, "release": m6r}))
    pm = r[6]["e6_crosscheck"]["precache_missing"]
    items.append(item(6, "c6_precache", "확인", "서비스 워커가 미리 받을 목록에 빠진 파일이 없다",
                      "결과 인용", None, pm == [], f"{rel(result_path(6))}: e6_crosscheck.precache_missing", {"missing": pm}))
    items.append(item(6, "c6_path_choice", "확인", "꼭 필요한 조건(웹 코드 재사용, 앱스토어, 푸시, 기기 기능, 심사 없는 설치)을 적고 그 조건을 모두 만족하는 길을 골랐다",
                      "사람 확인", None, None, f"{rel(result_path(6))}: e5_paths.cases",
                      {"case_E_fits": r[6]["e5_paths"]["cases"]["E 웹 코드 그대로 + iOS 블루투스 + 심사 없이"]["fits"]},
                      "조건을 모두 만족하는 길이 없는 경우(6장 사례 E)도 있어요"))

    # 7장 포장·서명
    af, ar = android_check(FIRST["android"]), android_check(RELEASE["android"])
    items.append(item(7, "c7_release_config", "막음", "Capacitor·안드로이드 설정에 개발용 값(server.url, cleartext, 디버깅 등) '막음' 항목이 0개다",
                      "다시 검사", af["blocking"] == 0, ar["blocking"] == 0, "webapp_ch07_config.check_project",
                      {"first": af["blocking"], "release": ar["blocking"]}))
    items.append(item(7, "c7_permissions", "확인", "쓰는 플러그인이 요구하지 않는 권한이 없다",
                      "다시 검사", af["permission_unused"] == 0, ar["permission_unused"] == 0, "webapp_ch07_config.check_project",
                      {"first": af["permission_unused"], "release": ar["permission_unused"]}))
    items.append(item(7, "c7_signing_machine", "막음", "서명 키 점검표의 기계 항목(키 파일 위치, 비밀번호, 유효기간, versionCode 등)을 모두 통과한다",
                      "다시 검사", af["signing_machine_pass"] == af["signing_machine_total"],
                      ar["signing_machine_pass"] == ar["signing_machine_total"], "webapp_ch07_signing.scorecard",
                      {"first": f"{af['signing_machine_pass']}/{af['signing_machine_total']}",
                       "release": f"{ar['signing_machine_pass']}/{ar['signing_machine_total']}"}))
    items.append(item(7, "c7_signing_human", "막음", "서명 키 사람 항목 4개(두 곳 백업, 지문 기록, 담당자, 릴리스 키 서명)에 답했다",
                      "사람 확인", None, None, "webapp_ch07_signing.ITEMS(사람)",
                      {"human_items": ar["signing_human_total"]}))

    # 8장 플랫폼·수수료 (판단이라 사람 확인, 8장 결과가 있으면 참고값을 덧붙임)
    refs = ch08_refs() or {}
    for row in json.loads(CH08_MANUAL.read_text(encoding="utf-8"))["items"]:
        value = {"facts": row.get("facts"), "ch08": refs.get(row["id"])}
        items.append(item(8, row["id"], row["gate"], row["question"], "사람 확인", None, None,
                          rel(CH08_MANUAL), value, row.get("note", "")))

    # 9장 남는 것 (운영)
    items.append(item(9, "c9_backup_restore", "확인", "사용자 카드·브랜드 설정을 백업하고, 실제로 한 번 되살려 봤다",
                      "사람 확인", None, None, "checks/ch09.csv", None, "[해석] 이 장에서 덧붙인 운영 항목"))
    items.append(item(9, "c9_support_channel", "확인", "문의를 받고 장애를 알릴 창구와, 키를 바꿔야 할 때 할 일을 적어 두었다",
                      "사람 확인", None, None, "checks/ch09.csv", None, "[해석] 이 장에서 덧붙인 운영 항목"))
    return items


def tally(items):
    auto = [i for i in items if i["how"] != "사람 확인"]
    with_first = [i for i in auto if i["first"] is not None]
    manual = [i for i in items if i["how"] == "사람 확인"]
    by_ch = {}
    for i in items:
        b = by_ch.setdefault(i["ch"], {"items": 0, "auto": 0, "manual": 0})
        b["items"] += 1
        b["auto" if i["how"] != "사람 확인" else "manual"] += 1
    return {
        "items": len(items),
        "recheck": sum(1 for i in items if i["how"] == "다시 검사"),
        "quoted": sum(1 for i in items if i["how"] == "결과 인용"),
        "auto": len(auto),
        "manual": len(manual),
        "release_pass": sum(1 for i in auto if i["release"] is True),
        "release_fail": [i["id"] for i in auto if i["release"] is False],
        "first_judged": len(with_first),
        "first_pass": sum(1 for i in with_first if i["first"] is True),
        "first_fail": [i["id"] for i in with_first if i["first"] is False],
        "first_fail_blocking": sum(1 for i in with_first if i["first"] is False and i["gate"] == "막음"),
        "gate_block": sum(1 for i in items if i["gate"] == "막음"),
        "manual_block": sum(1 for i in manual if i["gate"] == "막음"),
        "by_ch": by_ch,
    }


def as_csv_rows(items):
    def mark(v):
        return {True: "통과", False: "고칠 것", None: "-"}[v]
    head = ["장", "id", "막음/확인", "점검", "확인 방법", "첫 판", "출시 후보", "근거"]
    rows = [head]
    for i in items:
        rel_mark = mark(i["release"]) if i["how"] != "사람 확인" else "사람이 답함"
        rows.append([f"{i['ch']}장", i["id"], i["gate"], i["question"], i["how"], mark(i["first"]), rel_mark, i["evidence"]])
    return rows


if __name__ == "__main__":
    its = build_items()
    for i in its:
        print(f"{i['ch']}장 {i['gate']} {i['how']:5} 첫판={i['first']!s:5} 후보={i['release']!s:5} {i['question']}")
    print(json.dumps(tally(its), ensure_ascii=False))
