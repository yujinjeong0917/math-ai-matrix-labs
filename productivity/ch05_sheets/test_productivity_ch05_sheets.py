"""생산성 도구 5장 검증. `uv run pytest -q productivity/ch05_sheets` 로 실행한다.

experiments 모듈은 장마다 이름이 같아 import하지 않는다. 필요한 값은 모듈에서 직접 다시 계산한다.
"""

import json
import random
from pathlib import Path

import productivity_ch05_data as DATA
import productivity_ch05_sheet as SH
import productivity_ch05_workbook as WB

HERE = Path(__file__).parent


def test_first_screen_hand_numbers():
    # 가 2/20 = 10%, 나 3/25 = 12%, 다 2/25 = 8%, 라 6/20 = 30%, 마 5/20 = 25%
    rates = [c / n for _, n, c in DATA.HAND]
    assert [round(r, 2) for r in rates] == [0.10, 0.12, 0.08, 0.30, 0.25]
    assert round(sum(rates[:3]) / 3, 4) == 0.10  # (10 + 12 + 8) / 3
    assert round(sum(rates) / 5, 4) == 0.17      # (10 + 12 + 8 + 30 + 25) / 5 = 85 / 5
    assert sum(rates[:3]) / 3 < DATA.DECISION_THRESHOLD <= sum(rates) / 5


def test_closed_range_does_not_grow_on_append_but_open_range_does():
    sh = SH.Sheet(["b", "n", "c", "r"], [["가", 20, 2, 0.1], ["나", 25, 3, 0.12], ["다", 25, 2, 0.08]])
    closed = sh.add_formula("AVERAGE", "D2:D4")
    opened = sh.add_formula("AVERAGE", "D2:D")
    sh.append_row(["라", 20, 6, 0.3])
    sh.append_row(["마", 20, 5, 0.25])
    assert closed[1].text() == "D2:D4"
    assert abs(sh.evaluate(closed) - 0.10) < 1e-12
    assert abs(sh.evaluate(opened) - 0.17) < 1e-12


def test_insert_inside_range_grows_closed_range():
    sh = SH.Sheet(["b", "r"], [["가", 0.1], ["나", 0.12], ["다", 0.08]])
    f = sh.add_formula("AVERAGE", "B2:B4")
    sh.insert_row(3, ["라", 0.3])
    assert f[1].text() == "B2:B5"


def test_open_range_picks_up_typed_total_row():
    sh = SH.Sheet(["b", "n"], [["가", 20], ["나", 25]])
    f = sh.add_formula("SUM", "B2:B")
    sh.append_row(["합계", 45])
    assert sh.evaluate(f) == 90


def test_save_capped_keeps_header_plus_65535_rows():
    lines = list(range(70_000))
    kept = SH.save_capped(lines)
    assert len(kept) == 65_535 and kept[-1] == 65_534


def test_auto_convert_and_plain_text():
    assert SH.auto_convert("2-03") == ("date", "2026-02-03")
    assert SH.auto_convert("13-05") == ("text", "13-05")
    assert SH.auto_convert("007") == ("number", 7)
    assert SH.paste_column(["2-03", "007"], plain_text=True) == [("text", "2-03"), ("text", "007")]


def test_every_raw_row_goes_to_exactly_one_place():
    rows, _ = DATA.make_bookings(seed=5)
    raw = WB.Raw(DATA.messy_copy(rows))
    kept, dropped = WB.clean(raw)
    assert len(kept) + len(dropped) == len(raw)
    assert {r["status"] for r in kept} <= set(DATA.STATUSES)


def test_clean_data_passes_all_checks():
    rows, _ = DATA.make_bookings(seed=5)
    raw = WB.Raw(rows)
    ck = WB.checks(raw, import_fingerprint=raw.fingerprint)
    assert WB.failed(ck) == []


def test_each_failure_trips_the_intended_check():
    rows, _ = DATA.make_bookings(seed=5)
    fp = WB.Raw(rows).fingerprint
    # 행 수 상한: 보낸 쪽은 더 많이 보냈다
    assert "C6 보낸 행 수 = 원본 행" in WB.failed(WB.checks(WB.Raw(rows[:-10], source_rows=len(rows))))
    # 날짜 '5일'
    bad = [dict(rows[0], date="5일")] + rows[1:]
    assert WB.failed(WB.checks(WB.Raw(bad))) == ["C4 날짜 변환 실패 0"]
    # 상태 오타
    bad = [dict(rows[0], status="취쇼")] + rows[1:]
    assert WB.failed(WB.checks(WB.Raw(bad))) == ["C3 표준 밖 상태 0"]
    # 합계 줄
    bad = rows + [dict(rows[0], booking_id="합계", branch="합계")]
    assert WB.failed(WB.checks(WB.Raw(bad))) == ["C7 목록 밖 지점 0"]
    # 열 하나만 정렬한 사본: C1~C7은 통과하고 지문만 바뀐다
    sts = sorted(r["status"] for r in rows)
    srt = [dict(r, status=s) for r, s in zip(rows, sts)]
    assert WB.failed(WB.checks(WB.Raw(srt))) == []
    assert WB.failed(WB.checks(WB.Raw(srt), import_fingerprint=fp)) == ["C8 원본 지문 그대로"]


def test_every_drop_reason_has_a_failing_check():
    """clean()이 행을 빼는 사유마다 FALSE가 되는 점검이 있어야 한다(조용히 빠지는 행 금지)."""
    rows, _ = DATA.make_bookings(seed=5)
    cases = {
        "상태 모름": dict(rows[0], status="취쇼"),
        "날짜 모름": dict(rows[0], date="5일"),
        "지점 모름": dict(rows[0], branch="B99"),
        "이용 시간 범위 밖": dict(rows[0], hours="13"),
    }
    assert set(cases) == set(WB.DROP_REASON_CHECK)
    for why, bad in cases.items():
        raw = WB.Raw([bad] + rows[1:])
        _, dropped = WB.clean(raw)
        assert [w for _, w in dropped] == [why]
        failed = WB.failed(WB.checks(raw))
        assert len(failed) == 1 and failed[0].startswith(WB.DROP_REASON_CHECK[why] + " ")


def test_valid_but_wrong_value_slips_through():
    rows, _ = DATA.make_bookings(seed=5)
    wrong = [dict(r, status="완료") if r["status"] == "노쇼" else r for r in rows]
    assert WB.failed(WB.checks(WB.Raw(wrong))) == []


def test_raw_is_protected():
    raw = WB.Raw([])
    try:
        raw.edit()
    except WB.ProtectedError:
        return
    raise AssertionError("raw는 고칠 수 없어야 한다")


def test_pivot_three_implementations_agree_and_conserve():
    rng = random.Random(1)
    for _ in range(100):
        rows = [{"branch": rng.choice(DATA.BRANCHES[:5]), "status": rng.choice(DATA.STATUSES)}
                for _ in range(rng.randint(1, 200))]
        a, b, c = WB.pivot_dict(rows), WB.pivot_sql(rows), WB.pivot_numpy(rows)
        assert a == b == c
        assert WB.pivot_total(a) == len(rows)


def test_unique_customers_are_not_additive():
    rows = [{"branch": "B01", "customer": "C1"}, {"branch": "B02", "customer": "C1"}, {"branch": "B02", "customer": "C2"}]
    per = {}
    for r in rows:
        per.setdefault(r["branch"], set()).add(r["customer"])
    assert sum(len(v) for v in per.values()) == 3 and len({r["customer"] for r in rows}) == 2


def test_results_file_matches_modules():
    res = json.loads((HERE / "results" / "ch05.json").read_text())
    rows, _ = DATA.make_bookings(seed=res["meta"]["seed"])
    assert res["meta"]["rows"] == len(rows)
    t = DATA.branch_table(rows)
    assert res["e1_range"]["closed_D2:D16"]["value"] == round(sum(x["rate"] for x in t[:15]) / 15, 4)
    assert res["e1_range"]["truth_mean_rate_20"] == round(sum(x["rate"] for x in t) / 20, 4)
    assert res["e9_correspondence"]["mismatches"] == 0
