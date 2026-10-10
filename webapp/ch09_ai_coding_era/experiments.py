"""1인 개발 앱 출시 실전 9장 실험. `uv run python webapp/ch09_ai_coding_era/experiments.py` 로 실행한다.

E0 점검표 손계산: 항목 수, 확인 방법별 수, 두 판의 통과 수
E1 점검표 전체(첫 판·출시 후보), results/checklist_ch09.csv 도 함께 쓴다
E2 사람만 답할 수 있는 항목을 무엇에 관한 것인지(돈, 운영, 선택, 데이터)로 나누기
E3 1장으로 돌아가기: 사용자가 볼 수 있는 바이트와 서버에만 남은 바이트(1장 결과 인용)
E4 시장 숫자: 같은 출처 안의 두 값끼리만 나눈 비율
E5 대조: 인용한 값이 지금의 앞 장 JSON과 같은가, 다시 검사한 값이 앞 장 결과와 같은가
E6 겹치는 점검: 2장 모양 점검기가 3장 첫 판의 가짜 AI 키를 찾는가(3장은 이름으로 셌어요)

표준 라이브러리만 쓴다. 서버를 띄우지 않고, 네트워크와 API 키 없이 돈다. 결과는 매번 같다.
"""

import csv
import json
import platform
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import webapp_ch09_checklist as L  # noqa: E402

OUT = HERE / "results" / "ch09.json"
CSV_OUT = HERE / "results" / "checklist_ch09.csv"
MARKET = HERE / "data" / "market_2026-10-10.json"

MANUAL_KIND = {
    "c4_month_cost": "돈", "c5_free_tier_month": "돈", "c8_store_fee": "돈", "c8_account_cost": "돈",
    "c5_pause": "운영", "c7_signing_human": "운영", "c9_support_channel": "운영", "c8_play_closed_test": "운영",
    "c6_path_choice": "선택", "c8_target_platform": "선택",
    "c9_backup_restore": "데이터",
}


def e2_manual(items):
    out = {}
    for i in items:
        if i["how"] == "사람 확인":
            out.setdefault(MANUAL_KIND[i["id"]], []).append(i["id"])
    return {k: {"count": len(v), "ids": v} for k, v in sorted(out.items(), key=lambda kv: -len(kv[1]))}


def e3_bookend():
    r1 = L.load_result(1)
    return {
        "visible_bytes": r1["e0_first_open"]["all_bytes"],
        "visible_responses": r1["e0_first_open"]["all_requests"],
        "server_only_bytes": r1["e2_summary"]["server_only_bytes"],
        "server_only_files": r1["e2_summary"]["server_only"],
        "markers_in_downloads": r1["e3_server_only_markers_in_downloads"],
    }


def e4_market():
    m = json.loads(MARKET.read_text(encoding="utf-8"))
    rc, so, pe = m["revenuecat_sosa_2026"], m["stackoverflow_2025"], m["pearce_2022"]
    return {
        "launch_growth": round(rc["new_sub_apps_per_month_2026_01"] / rc["new_sub_apps_per_month_2022_01"], 2),
        "top10_over_median": round(rc["top10_monthly_revenue_1y_usd"] / rc["median_monthly_revenue_1y_usd"], 1),
        "share_1k_to_10k_kept": round(rc["share_hit_10k_all"] / rc["share_hit_1k_all"], 3),
        "pearce_vulnerable_programs_approx": round(pe["programs"] * pe["vulnerable_share_approx"]),
        "so_ai_use_change_pt": round((so["use_or_plan_ai_2025"] - so["use_or_plan_ai_2024"]) * 100),
        "so_distrust_change_pt": round((so["distrust_accuracy_2025"] - so["distrust_accuracy_2024"]) * 100),
    }


def e5_crosscheck(items):
    by = {i["id"]: i for i in items}
    r = {n: L.load_result(n) for n in L.CH}
    checks = {}
    # 인용: 3장 다섯 항목이 지금 JSON과 같은가
    rows = r[3]["e0_checklist"]["items"]
    checks["ch03_quoted_equal"] = all(
        by[iid]["first"] == L.get(r[3], f"e0_checklist.items[{k}].before_pass")
        and by[iid]["release"] == L.get(r[3], f"e0_checklist.items[{k}].after_pass")
        for k, iid in enumerate(["c3_rls", "c3_ai_key", "c3_admin", "c3_upload", "c3_rate_limit"])) and len(rows) == 5
    # 다시 검사: 7장 결과와 같은가
    checks["ch07_blocking_equal"] = (by["c7_release_config"]["value"]["first"] == r[7]["e1_checker"]["cardnews_mistake"]["blocking"]
                                     and by["c7_release_config"]["value"]["release"] == r[7]["e1_checker"]["cardnews_good"]["blocking"])
    checks["ch07_permissions_equal"] = (by["c7_permissions"]["value"]["first"] == r[7]["e2_permissions"]["cardnews_mistake"]["flagged"])
    checks["ch07_signing_equal"] = (
        by["c7_signing_machine"]["value"]["first"] == "{machine_pass}/{machine_total}".format(**r[7]["e3_signing"]["cardnews_mistake"])
        and by["c7_signing_machine"]["value"]["release"] == "{machine_pass}/{machine_total}".format(**r[7]["e3_signing"]["cardnews_good"]))
    # 다시 검사: 6장 결과와 같은가
    checks["ch06_manifest_equal"] = (by["c6_manifest"]["value"]["first"]["installable"] == r[6]["e1_manifests"]["broken_first_draft"]["chromium_installable"]
                                     and by["c6_manifest"]["value"]["release"]["installable"] == r[6]["e1_manifests"]["ours"]["chromium_installable"])
    checks["ch06_no_sourcemap_line_equal"] = (r[6]["e6_crosscheck"]["sourcemap_line_in_app_min_js"] is False
                                              and by["c1_source_map"]["release"] is True)
    # 다시 검사: 2장 결과와 같은가(dist_mistake 비밀 키 2개, dist_good 0개)
    kf = {x["folder"]: x for x in by["c2_secret_in_build"]["value"]["first"]}
    kr = {x["folder"]: x for x in by["c2_secret_in_build"]["value"]["release"]}
    checks["ch02_scan_equal"] = (kf["ch02_api_keys/project/dist_mistake"]["secret"] == r[2]["e2_builds"]["mistake"]["scan"]["demo"]["summary"]["secret"]
                                 and kr["ch02_api_keys/project/dist_good"]["secret"] == r[2]["e2_builds"]["good"]["scan"]["demo"]["summary"]["secret"])
    # 다시 검사: 1장 결과와 같은가(서버에만 둔 값 0번)
    checks["ch01_markers_equal"] = by["c1_server_only"]["value"]["first"] == r[1]["e3_server_only_markers_in_downloads"]
    checks["all"] = all(checks.values())
    return checks


def e6_overlap(items):
    by = {i["id"]: i for i in items}
    r3 = L.load_result(3)
    scan = {x["folder"]: x for x in by["c2_secret_in_build"]["value"]["first"]}["ch03_vibe_security/app/public_before"]
    return {"ch02_scanner_secret_found": scan["secret"],
            "ch03_counted_ai_secret_key": r3["e2_keys"]["before"]["ai_secret_key_count"],
            "scanner_missed": r3["e2_keys"]["before"]["ai_secret_key_count"] - scan["secret"],
            "caught_by_other_item": by["c3_ai_key"]["first"] is False}


def main():
    items = L.build_items()
    t = L.tally(items)
    res = {
        "meta": {"python": platform.python_version(), "platform": f"{platform.system()} {platform.machine()}",
                 "checked": "2026-10-10",
                 "read_results": {L.rel(L.result_path(n)): L.sha256(L.result_path(n)) for n in L.CH},
                 "ch08_result": ({L.rel(L.CH08_RESULT): L.sha256(L.CH08_RESULT)} if L.CH08_RESULT.is_file() else None),
                 "ch08": "8장 항목은 판단이라 사람 확인. 공식 문서 값은 data/ch08_manual_2026-10-10.json, 8장 계산 값은 참고값으로 인용"},
        "e0_tally": t,
        "e1_items": items,
        "e2_manual_by_kind": e2_manual(items),
        "e3_bookend_ch01": e3_bookend(),
        "e4_market_ratios": e4_market(),
        "e5_crosscheck": e5_crosscheck(items),
        "e6_overlap": e6_overlap(items),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(L.as_csv_rows(items))
    print(json.dumps({k: res[k] for k in ("e0_tally", "e2_manual_by_kind", "e3_bookend_ch01", "e4_market_ratios", "e5_crosscheck", "e6_overlap")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
