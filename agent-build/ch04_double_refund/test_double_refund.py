import json
from pathlib import Path

import pytest

from double_refund import RefundServer, refund_amounts, run_variant, simulate

RESULTS = Path(__file__).resolve().parent / "results" / "ch04.json"


def test_correct_total_is_330():
    assert len(refund_amounts) == 10
    assert sum(refund_amounts.values()) == 330


@pytest.mark.parametrize("variant, expected", [
    ("A", (13, 420, 90)),
    ("B", (11, 355, 25)),
    ("C", (10, 330, 0)),
    ("D", (13, 420, 90)),
])
def test_variant_numbers_match_chapter_table(variant, expected):
    assert run_variant(variant) == expected


def doubled_orders(variant):
    orders = [o for o, _ in simulate(variant).payouts]
    return sorted({o for o in orders if orders.count(o) > 1})


def test_which_orders_leak():
    assert doubled_orders("A") == [3, 5, 7]
    assert doubled_orders("B") == [5]
    assert doubled_orders("C") == []
    assert doubled_orders("D") == [3, 5, 7]


def test_extra_amount_is_sum_of_doubled_orders():
    assert refund_amounts[3] + refund_amounts[7] + refund_amounts[5] == 90
    assert refund_amounts[5] == 25


def test_same_key_returns_first_result_without_new_payout():
    server = RefundServer({1: 30})
    assert server.refund(1, "refund-1") == "ok"
    assert server.refund(1, "refund-1") == "ok"
    assert server.payouts == [(1, 30)]


def test_late_reply_still_pays_once():
    server = RefundServer({1: 30})
    with pytest.raises(TimeoutError):
        server.refund(1, "refund-1", reply_is_late=True)
    assert server.payouts == [(1, 30)]
    assert server.order_status[1] == "환불됨"


def test_results_json_matches_code():
    saved = json.loads(RESULTS.read_text(encoding="utf-8"))["variants"]
    for v in "ABCD":
        count, total, extra = run_variant(v)
        assert (saved[v]["refund_count"], saved[v]["total_paid"], saved[v]["extra_over_correct"]) == (count, total, extra)
    assert [saved[v]["blocked_of_planted"] for v in "ABCD"] == [0, 2, 3, 0]
