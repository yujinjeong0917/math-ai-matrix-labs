"""생산성 도구 5장 비교 실험. `uv run python productivity/ch05_sheets/experiments.py` 로 실행한다.

E0 첫 화면 손계산: 지점 5곳 취소율, AVERAGE 범위 3칸 vs 5칸
E1 범위 누락: 지점 15곳일 때 쓴 AVERAGE(D2:D16)에 지점 5곳을 아래에 붙이면 / 열린 범위 / 맨 아래 합계 줄
E2 범위 누락이 결정을 뒤집는 빈도: 시드 1000개, 새 지점이 다를 때와 같을 때
E3 행 수 상한: 이벤트 로그를 65,536행 형식으로 저장하면
E4 원본 덮어쓰기: 상태 열 하나만 정렬, 행 삭제
E5 입력 혼재: 날짜·상태 표기가 섞이면 피벗이 몇 줄로 쪼개지나
E6 자동 형식 변환: 방 번호 '2-03'이 날짜로
E7 명세가 못 잡는 것: 목록 안의 틀린 값(노쇼를 완료로)
E8 고유 고객 수는 더할 수 없다
E9 대응 구현 대조: 무작위 데이터 500개에서 피벗 dict / sqlite3 / NumPy와 보존식
E10 재계산 시간과 셀 한도
S  요약: 실패마다 자유 시트 vs 명세 통합 문서

고정 시드라 E10의 시간 말고는 결과가 매번 같다.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import csv  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).parent
# 같은 폴더의 모듈(productivity_chNN_*)을 어느 디렉터리에서 실행해도 찾도록 경로를 앞에 넣는다
sys.path.insert(0, str(HERE))

import productivity_ch05_data as DATA  # noqa: E402
import productivity_ch05_sheet as SH  # noqa: E402
import productivity_ch05_workbook as WB  # noqa: E402

OUT = HERE / "results" / "ch05.json"
SEED = 5  # 합성 예약 기록(make_bookings)의 기본 시드. 대부분의 실험이 이 데이터를 공유한다
# R4: None은 그대로 두고 소수 넷째 자리로 반올림(JSON에 저장할 값의 표기를 통일)
R4 = lambda x: None if x is None else round(x, 4)  # noqa: E731


# ---------------------------------------------------------------- E0
def e0_hand():
    """E0: 첫 화면 손계산. 지점 5곳 중 앞 3곳만 범위(D2:D4)에 든 평균(10%)과 5곳 전체 평균(17%)이 15% 기준의 어느 쪽인지(결정이 뒤집히나)."""
    rates = [c / n for _, n, c in DATA.HAND]
    sh = SH.Sheet(["지점", "예약", "취소", "취소율"], [[b, n, c, c / n] for b, n, c in DATA.HAND[:3]])
    f = sh.add_formula("AVERAGE", "D2:D4")
    for b, n, c in DATA.HAND[3:]:
        sh.append_row([b, n, c, c / n])
    narrow = sh.evaluate(f)
    full = sum(rates) / len(rates)
    return {
        "rows": [{"branch": b, "bookings": n, "cancels": c, "rate": R4(c / n)} for b, n, c in DATA.HAND],
        "formula_after_append": f[1].text(),
        "avg_narrow": R4(narrow),
        "avg_full": R4(full),
        "threshold": DATA.DECISION_THRESHOLD,
        "decision_narrow": narrow >= DATA.DECISION_THRESHOLD,
        "decision_full": full >= DATA.DECISION_THRESHOLD,
    }


# ---------------------------------------------------------------- E1
def branch_sheet(table, upto):
    """지점 표의 앞 upto곳으로 시트를 만든다(나머지 지점은 이후 append로 붙이는 시나리오)."""
    return SH.Sheet(["branch", "bookings", "cancels", "rate"],
                    [[t["branch"], t["bookings"], t["cancels"], t["rate"]] for t in table[:upto]])


def e1_range(rows):
    """E1: 15곳으로 시트와 수식을 만들고 5곳을 아래에 붙였을 때 닫힌 범위 / 열린 범위 / 안쪽에 끼워 넣기의 값, 열린 범위에 합계 줄이 섞일 때의 SUM, 명세 통합 문서의 같은 지표."""
    table = DATA.branch_table(rows)
    truth = sum(t["rate"] for t in table) / len(table)
    out = {"truth_mean_rate_20": R4(truth)}
    # 런칭 때 15곳으로 시트를 만들고 수식을 썼다. 둘째 주에 5곳을 아래에 붙인다.
    for label, rng_text in [("closed_D2:D16", "D2:D16"), ("open_D2:D", "D2:D")]:
        sh = branch_sheet(table, 15)
        f = sh.add_formula("AVERAGE", rng_text)
        for t in table[15:]:
            sh.append_row([t["branch"], t["bookings"], t["cancels"], t["rate"]])
        v = sh.evaluate(f)
        out[label] = {"formula_after": f[1].text(), "value": R4(v), "n_in_range": len(sh.column("D", f[1])),
                      "decision": v >= DATA.DECISION_THRESHOLD}
    # 대조: 범위 안쪽(B10 다음)에 끼워 넣으면 닫힌 범위도 늘어난다(모형 가정)
    sh = branch_sheet(table, 15)
    f = sh.add_formula("AVERAGE", "D2:D16")
    for k, t in enumerate(table[15:]):
        sh.insert_row(12 + k, [t["branch"], t["bookings"], t["cancels"], t["rate"]])
    out["closed_insert_inside"] = {"formula_after": f[1].text(), "value": R4(sh.evaluate(f))}
    out["truth_decision"] = truth >= DATA.DECISION_THRESHOLD
    # 열린 범위의 대가: 누가 맨 아래에 '합계' 줄을 손으로 적으면
    sh = branch_sheet(table, 20)
    fs = sh.add_formula("SUM", "B2:B")
    before = sh.evaluate(fs)
    sh.append_row(["합계", before, sum(t["cancels"] for t in table), None])
    out["open_range_total_row"] = {"sum_before": before, "sum_after": sh.evaluate(fs),
                                   "ratio": sh.evaluate(fs) / before}
    # 같은 데이터를 명세 통합 문서로: clean 지점별 취소율의 평균(지점 목록은 lookup)
    raw = WB.Raw(rows)
    kept, _ = WB.clean(raw)
    m = WB.metrics(kept)
    out["spec_workbook"] = {"value": R4(m["mean_branch_cancel_rate"]), "n_branches": m["n_branches_in_metric"],
                            "pooled_rate": R4(m["pooled_cancel_rate"])}
    # 합계 줄이 raw에 섞여 들어오면 C7이 잡는다
    stray = rows + [{"booking_id": "합계", "date": "2026-10-04", "branch": "합계", "room": "", "hours": "1",
                     "status": "완료", "customer": ""}]
    out["spec_total_row_check"] = WB.failed(WB.checks(WB.Raw(stray)))
    out["first15_mean"] = R4(sum(t["rate"] for t in table[:15]) / 15)
    out["last5_mean"] = R4(sum(t["rate"] for t in table[15:]) / 5)
    return out


# ---------------------------------------------------------------- E2
def e2_flip(n=1000):
    """E2: 시드 n개(10000부터)에서 15곳 평균과 20곳 평균이 15% 기준의 서로 다른 쪽에 놓여 결정이 뒤집히는 횟수. 새 지점이 다를 때(wave_gap=True)와 같을 때를 비교한다."""
    out = {}
    for gap in (True, False):
        flips, diffs = 0, []
        for s in range(n):
            rows, _ = DATA.make_bookings(seed=10_000 + s, wave_gap=gap)
            t = DATA.branch_table(rows)
            a15 = sum(x["rate"] for x in t[:15]) / 15
            a20 = sum(x["rate"] for x in t) / 20
            diffs.append(a20 - a15)  # 20곳 평균 - 15곳 평균(비율 차이, 아래에서 %p로 환산)
            flips += (a15 >= DATA.DECISION_THRESHOLD) != (a20 >= DATA.DECISION_THRESHOLD)
        d = np.array(diffs)
        out["new_branches_differ" if gap else "new_branches_same"] = {
            "n": n, "decision_flips": flips, "flip_rate": R4(flips / n),
            "mean_diff_pp": round(float(d.mean()) * 100, 2), "max_abs_diff_pp": round(float(np.abs(d).max()) * 100, 2)}
    return out


# ---------------------------------------------------------------- E3
def e3_row_cap():
    """E3: 예약 건수 1천~10만 건의 이벤트 로그(예약당 3줄)를 65,536행 형식으로 저장할 때 사라지는 줄, 아예 없어진 예약, 일부만 남은 예약, 마지막으로 남은 날짜."""
    out = []
    for nb in (1_000, 10_000, 25_000, 100_000):
        log = DATA.event_log(nb)
        kept = SH.save_capped(log)
        lost_rows = len(log) - len(kept)
        ids_full = {r[0] for r in log}
        ids_kept = {r[0] for r in kept}
        # 이벤트 3줄이 모두 남은 예약만 온전하다
        cnt = {}
        for r in kept:
            cnt[r[0]] = cnt.get(r[0], 0) + 1
        whole = sum(1 for v in cnt.values() if v == 3)
        days_full = sorted({r[1] for r in log})
        days_kept = sorted({r[1] for r in kept})
        last_kept = kept[-1][1] if kept else None
        # 명세: 보낸 쪽 행 수(3 x 예약 수)를 함께 받고 C6으로 비교
        c6 = len(log) == len(kept)  # 보낸 행 수와 저장된 행 수가 같으면 통과(잘렸으면 실패 = 점검이 잡는다)
        out.append({
            "bookings": nb, "rows": len(log), "rows_saved": len(kept), "rows_lost": lost_rows,
            "bookings_missing_entirely": len(ids_full - ids_kept),
            "bookings_incomplete": len(ids_full) - whole,
            "days_total": len(days_full), "days_fully_kept": sum(1 for d in days_full if d < (last_kept or "")),
            "last_day_seen": last_kept, "c6_pass": c6,
            # 비교용: 신형 형식의 행 상한 1,048,576행(머리글 포함, 값은 Microsoft 문서 기준이며 확인 필요)에서는 얼마나 잘리나
            "xlsx_cap_rows_lost": max(0, len(log) - (1_048_576 - 1)),
        })
    return {"cap_rows_incl_header": SH.XLS_MAX_ROWS, "events_per_booking": 3, "runs": out}


# ---------------------------------------------------------------- E4
def e4_overwrite(rows):
    """E4: 상태 열 하나만 정렬(다른 열은 그대로)했을 때 지점별 취소율 변화, 원본 보호와 지문 점검, 행 50개를 지웠을 때 점검이 잡는 것."""
    truth = WB.branch_rates(rows)
    # 자유 시트: 상태 열 하나만 골라 정렬(다른 열은 그대로)
    rng = random.Random(SEED)
    sts = sorted(r["status"] for r in rows)  # 상태 값만 사전순으로 정렬해 원래 행과 어긋나게 다시 붙인다
    bad = [dict(r, status=s) for r, s in zip(rows, sts)]
    after = WB.branch_rates(bad)
    diffs = [abs(after[b] - truth[b]) for b in DATA.BRANCHES]
    pooled_before = sum(r["status"] == "취소" for r in rows) / len(rows)
    pooled_after = sum(r["status"] == "취소" for r in bad) / len(bad)
    # 명세 통합 문서: raw는 고칠 수 없다. 누군가 사본에서 정렬해 다시 붙여 넣으면 지문이 바뀐다.
    raw = WB.Raw(rows)
    try:
        raw.edit()
        blocked = False
    except WB.ProtectedError:
        blocked = True
    # 보낸 행 수(source_rows)는 맞추고 지문(import_fingerprint)만 비교한다: 정렬은 행 수·합계를 안 바꾸므로 지문(C8)만 잡는다
    ck_sorted = WB.failed(WB.checks(WB.Raw(bad, source_rows=len(rows)), import_fingerprint=raw.fingerprint))
    # 행 삭제: 자유 시트에서 50행을 지우면 원래 몇 행이었는지 남지 않는다
    idx = sorted(rng.sample(range(len(rows)), 50))
    gone = set(idx)
    deleted = [r for i, r in enumerate(rows) if i not in gone]
    ck_deleted = WB.failed(WB.checks(WB.Raw(deleted, source_rows=len(rows))))
    # 같은 범위에서 열 하나만 정렬해도 합계는 그대로다(C2가 못 잡는 이유)
    ck_sorted_no_fp = WB.failed(WB.checks(WB.Raw(bad, source_rows=len(rows))))
    return {
        "rows": len(rows),
        "sort_one_column": {
            "max_abs_branch_rate_change_pp": round(max(diffs) * 100, 2),
            "branches_changed_over_5pp": sum(d > 0.05 for d in diffs),
            "pooled_cancel_before": R4(pooled_before), "pooled_cancel_after": R4(pooled_after),
            "spec_raw_edit_blocked": blocked,
            "spec_checks_failed_with_fingerprint": ck_sorted,
            "spec_checks_failed_without_fingerprint": ck_sorted_no_fp,
        },
        "delete_50_rows": {"rows_after": len(deleted), "spec_checks_failed": ck_deleted},
    }


# ---------------------------------------------------------------- E5
def e5_messy(rows):
    """E5: 날짜·상태 표기가 섞인 사본(messy_copy)을 자유 시트처럼 그대로 피벗할 때와 명세 통합 문서로 정리할 때의 차이(쪼개진 상태 줄, 못 고치는 "N일" 날짜 행)."""
    messy = DATA.messy_copy(rows)
    # 자유 시트: 상태 글자 그대로 피벗
    status_labels = sorted({r["status"] for r in messy})
    cancel_exact = sum(r["status"] == "취소" for r in messy)
    cancel_true = sum(r["status"] == "취소" for r in rows)
    date_labels = len({r["date"] for r in messy})
    day_only = sum(r["date"].endswith("일") for r in messy)
    md = sum("/" in r["date"] for r in messy)
    # 명세 통합 문서
    raw = WB.Raw(messy)
    kept, dropped = WB.clean(raw)
    ck = WB.checks(raw)
    m_true = WB.metrics(rows)
    m_clean = WB.metrics(kept)
    return {
        "rows": len(messy),
        "free_pivot_status_rows": len(status_labels), "free_status_labels": status_labels,
        "free_cancel_count_exact_label": cancel_exact, "true_cancel_count": cancel_true,
        "free_pooled_cancel_rate": R4(cancel_exact / len(messy)), "true_pooled_cancel_rate": R4(cancel_true / len(rows)),
        "free_distinct_date_labels": date_labels, "true_days": DATA.DAYS,
        "rows_md_format": md, "rows_day_only": day_only,
        "spec_kept": len(kept), "spec_dropped": len(dropped),
        "spec_drop_reasons": dict(sorted({w: sum(1 for _, x in dropped if x == w) for _, w in dropped}.items())),
        "spec_checks_failed": WB.failed(ck),
        "spec_pooled_cancel_rate": R4(m_clean["pooled_cancel_rate"]),
        "spec_mean_branch_rate": R4(m_clean["mean_branch_cancel_rate"]),
        "true_mean_branch_rate": R4(m_true["mean_branch_cancel_rate"]),
    }


# ---------------------------------------------------------------- E6
def e6_autoconvert(rows):
    """E6: 방 번호 열을 그대로 붙여 넣으면 몇 칸이 날짜로 바뀌는지, 일반 텍스트 형식으로 붙이면 0칸인지."""
    rooms = [r["room"] for r in rows]
    free = SH.paste_column(rooms)
    safe = SH.paste_column(rooms, plain_text=True)
    distinct = sorted(set(rooms))
    conv = {v: SH.auto_convert(v) for v in distinct}
    examples = {k: v for k, v in list(conv.items())[:4]}
    return {
        "rows": len(rooms), "distinct_rooms": len(distinct),
        "free_converted_to_date": sum(1 for t, _ in free if t == "date"),
        "plain_text_converted": sum(1 for t, _ in safe if t != "text"),
        "examples": {k: list(v) for k, v in examples.items()},
        "other_examples": {s: list(SH.auto_convert(s)) for s in ["007", "B-12", "13-05", "10/5"]},
    }


# ---------------------------------------------------------------- E7
def e7_valid_but_wrong(rows, q=0.3, seed=SEED):
    """E7: 노쇼 행의 30%를 "완료"로 잘못 기록한다. 표준 상태값 안의 틀린 값이라 점검이 하나도 실패하지 않지만 노쇼율이 달라진다."""
    rng = random.Random(seed)
    wrong = [dict(r, status="완료") if r["status"] == "노쇼" and rng.random() < q else r for r in rows]
    changed = sum(a["status"] != b["status"] for a, b in zip(rows, wrong))
    mt = WB.metrics(rows)
    kept, _ = WB.clean(WB.Raw(wrong))
    mw = WB.metrics(kept)
    return {"q": q, "rows_changed": changed, "true_noshow_rate": R4(mt["noshow_rate"]),
            "recorded_noshow_rate": R4(mw["noshow_rate"]),
            "spec_checks_failed": WB.failed(WB.checks(WB.Raw(wrong)))}


# ---------------------------------------------------------------- E8
def e8_unique(rows):
    """E8: 지점별 고유 고객 수의 합은 전체 고유 고객 수보다 크다(같은 고객이 여러 지점을 쓰므로 고유 수는 더할 수 없다)."""
    per = {}
    for r in rows:
        per.setdefault(r["branch"], set()).add(r["customer"])
    s = sum(len(v) for v in per.values())
    overall = len({r["customer"] for r in rows})
    return {"sum_of_branch_unique_customers": s, "overall_unique_customers": overall, "ratio": round(s / overall, 2),
            "rows": len(rows)}


# ---------------------------------------------------------------- E9
def e9_correspondence(n=500):
    """E9: 무작위 데이터 n개에서 피벗 세 구현(dict, sqlite3, NumPy)이 어긋난 횟수와 피벗 합계가 행 수와 같은 횟수(보존식)."""
    rng = random.Random(99)
    mism, cons = 0, 0
    for _ in range(n):
        k = rng.randint(1, 400)
        rows = [{"branch": rng.choice(DATA.BRANCHES[: rng.randint(1, 20)]), "status": rng.choice(DATA.STATUSES)}
                for _ in range(k)]
        a, b, c = WB.pivot_dict(rows), WB.pivot_sql(rows), WB.pivot_numpy(rows)
        mism += not (a == b == c)
        cons += WB.pivot_total(a) == len(rows)
    return {"datasets": n, "mismatches": mism, "conservation_holds": cons}


# ---------------------------------------------------------------- E10
def e10_timing():
    """E10: 행 수를 늘려 가며 raw 붙여 넣기 + checks의 소요 시간 중앙값(3회)과 필요한 칸 수. 시간은 기계마다 달라 결과가 매번 같지 않다."""
    out = []
    base, _ = DATA.make_bookings(seed=SEED)
    for n in (1_000, 10_000, 100_000):
        rows = [dict(base[i % len(base)], booking_id=f"X{i:06d}") for i in range(n)]
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            raw = WB.Raw(rows)
            WB.checks(raw)
            ts.append(time.perf_counter() - t0)
        cells = n * (len(WB.COLS) + len(WB.COLS) + 1)  # raw 7칸 + clean 7칸 + 제외 사유 1칸
        out.append({"rows": n, "seconds_median": round(sorted(ts)[1], 3), "cells_raw_plus_clean": cells})
    per_row = len(WB.COLS) * 2 + 1
    # 20,000,000칸은 Google Sheets 문서의 칸 수 한도(공식 문서 기준으로 적은 값, 확인 필요)다. 한 행이 15칸이라 한도 안에서 행 수 상한을 계산한다
    return {"runs": out, "sheets_cell_limit": 20_000_000, "cells_per_row": per_row,
            "max_rows_under_limit": 20_000_000 // per_row}


# ---------------------------------------------------------------- 요약
def summary(e1, e3, e4, e5, e6, e7):
    """실패 사례별로 "자유 시트는 못 잡음"과 "명세 통합 문서가 막거나 잡는 것"을 한 줄씩 모은 요약 표(장 본문의 비교 표용 자료)."""
    r25 = next(r for r in e3["runs"] if r["bookings"] == 25_000)
    return [
        {"failure": "범위 누락(아래에 붙인 지점 5곳)", "free": "못 잡음(값만 바뀜)",
         "spec": f"막음: clean 전체 열·lookup 지점 목록, 지점 {e1['spec_workbook']['n_branches']}곳"},
        {"failure": "행 수 상한(65,536행 형식)", "free": f"못 잡음({r25['rows_lost']}행 사라짐)",
         "spec": "잡음: C6" if not r25["c6_pass"] else "못 잡음"},
        {"failure": "원본 덮어쓰기(열 하나 정렬)", "free": "못 잡음(합계 그대로)",
         "spec": "막음: 보호 / 사본은 " + ",".join(e4["sort_one_column"]["spec_checks_failed_with_fingerprint"])},
        {"failure": "원본 덮어쓰기(행 삭제)", "free": "못 잡음",
         "spec": "잡음: " + ",".join(e4["delete_50_rows"]["spec_checks_failed"])},
        {"failure": "입력 혼재(날짜·상태 표기)", "free": f"못 잡음(상태 {e5['free_pivot_status_rows']}줄)",
         "spec": "고침 + 잡음: " + ",".join(e5["spec_checks_failed"])},
        {"failure": "자동 형식 변환(방 번호)", "free": f"못 잡음({e6['free_converted_to_date']}칸 날짜로)",
         "spec": f"막음: 일반 텍스트 형식({e6['plain_text_converted']}칸 변환)"},
        {"failure": "열린 범위 아래의 합계 줄", "free": f"못 잡음(SUM {e1['open_range_total_row']['ratio']:.0f}배)",
         "spec": "잡음: " + ",".join(e1["spec_total_row_check"])},
        {"failure": "목록 안의 틀린 값(노쇼→완료)", "free": "못 잡음",
         "spec": "못 잡음" if not e7["spec_checks_failed"] else ",".join(e7["spec_checks_failed"])},
    ]


def write_data(rows):
    """실습용 CSV 세 개(bookings, bookings_messy, branches)를 data/에 쓴다. 독자가 직접 붙여 넣어 보는 용도다."""
    (HERE / "data").mkdir(exist_ok=True)
    with open(HERE / "data" / "bookings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=WB.COLS)
        w.writeheader()
        w.writerows(rows)
    with open(HERE / "data" / "bookings_messy.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=WB.COLS)
        w.writeheader()
        w.writerows(DATA.messy_copy(rows))
    with open(HERE / "data" / "branches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["branch", "wave", "rooms"])
        for b in DATA.BRANCHES:
            w.writerow([b, 1 if b in DATA.FIRST_WAVE else 2, " ".join(DATA.rooms_of(b))])


def main():
    """데이터를 만들고 E0~E10을 돌려 results/ch05.json에 저장한다. E10의 시간 말고는 시드가 고정이라 결과가 매번 같다."""
    rows, _ = DATA.make_bookings(seed=SEED)
    write_data(rows)
    e1 = e1_range(rows)
    e3 = e3_row_cap()
    e4 = e4_overwrite(rows)
    e5 = e5_messy(rows)
    e6 = e6_autoconvert(rows)
    e7 = e7_valid_but_wrong(rows)
    res = {
        "meta": {"python": platform.python_version(), "numpy": np.__version__,
                 "platform": f"{platform.system()} {platform.machine()}", "seed": SEED, "checked": "2026-10-08",
                 "rows": len(rows), "branches": len(DATA.BRANCHES), "days": DATA.DAYS,
                 "threshold": DATA.DECISION_THRESHOLD},
        "e0_hand": e0_hand(),
        "e1_range": e1,
        "e2_flip": e2_flip(),
        "e3_row_cap": e3,
        "e4_overwrite": e4,
        "e5_messy": e5,
        "e6_autoconvert": e6,
        "e7_valid_but_wrong": e7,
        "e8_unique": e8_unique(rows),
        "e9_correspondence": e9_correspondence(),
        "e10_timing": e10_timing(),
    }
    res["summary"] = summary(e1, e3, e4, e5, e6, e7)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
