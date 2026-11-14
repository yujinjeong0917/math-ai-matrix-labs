import unittest

from context_limit import (BUDGET, HEADER, POLICY, apply_strategy, build_context, fake_agent_decisions,
                           first_overflow_turn, lost_clauses, run, total, turn_items)


class ContextLimitTest(unittest.TestCase):
    def test_policy_and_turn_sizes_by_hand(self):
        self.assertEqual([len(t) for _, _, t in POLICY], [126, 108, 130, 110, 104, 95, 112, 102])
        self.assertEqual(sum(len(t) for _, _, t in POLICY), 887)
        self.assertEqual(len(HEADER), 46)
        per_turn = [sum(s for _, _, s in turn_items(i)) for i in range(1, 8)]
        self.assertEqual(per_turn, [755, 850, 690, 745, 840, 680, 775])
        self.assertEqual(total(build_context(6)), 46 + 887 + sum(per_turn[:6]))
        self.assertEqual(total(build_context(6)), 5493)
        self.assertEqual(total(build_context(7)), 6268)

    def test_first_overflow(self):
        self.assertEqual(first_overflow_turn(), 7)
        self.assertLessEqual(total(build_context(6)), BUDGET)
        self.assertGreater(total(build_context(7)), BUDGET)

    def test_s1_loses_policy_front_first(self):
        r = run(7, "S1")
        self.assertEqual(r["overflow_chars"], 268)
        self.assertEqual(r["lost_clauses"], ["P1", "P2"])
        self.assertTrue(r["lost_header"])
        self.assertEqual(r["wrong_decisions"], 2)
        self.assertEqual(r["dropped_user_messages"], 0)
        # 잘려 나간 글자는 머리말 46 + P1 126 + P2 108 = 280으로 초과분 268을 처음 덮는다.
        self.assertEqual(46 + 126 + 108, 280)
        self.assertEqual(total(apply_strategy(build_context(7), "S1")), 6268 - 280)

    def test_s1_loses_every_clause_by_turn_8(self):
        self.assertEqual(run(8, "S1")["lost_clauses"], [p[0] for p in POLICY])
        self.assertEqual(run(8, "S1")["wrong_decisions"], 8)
        self.assertEqual(run(8, "S1")["dropped_user_messages"], 1)

    def test_pinned_policy_survives_but_conversation_is_cut(self):
        for n in (7, 8, 12, 20, 30):
            r = run(n, "S2")
            self.assertEqual(r["lost_clauses"], [], n)
            self.assertFalse(r["lost_header"])
            self.assertLessEqual(r["chars_after"], BUDGET)
        self.assertEqual(run(7, "S2")["dropped_user_messages"], 1)
        self.assertEqual(run(30, "S2")["dropped_user_messages"], 24)

    def test_clearing_tool_results_delays_cutting_conversation(self):
        self.assertEqual(run(12, "S3")["dropped_user_messages"], 0)
        self.assertEqual(run(12, "S3")["kept_full_tool_results"], 2)
        self.assertEqual(run(15, "S3")["dropped_user_messages"], 2)
        self.assertEqual(run(6, "S3"), {**run(6, "S2"), "strategy": "S3"})   # 넘지 않으면 아무것도 안 한다

    def test_all_strategies_fit_budget(self):
        for s in ("S1", "S2", "S3"):
            for n in (7, 12, 30):
                self.assertLessEqual(run(n, s)["chars_after"], BUDGET)

    def test_fake_agent_decisions(self):
        items = apply_strategy(build_context(7), "S1")
        d = fake_agent_decisions(items)
        self.assertEqual([k for k, v in d.items() if v != "올바름"], ["P1", "P2"])
        self.assertEqual(lost_clauses(items), ["P1", "P2"])

    def test_pinned_items_alone_over_budget_is_an_error(self):
        with self.assertRaises(ValueError):
            apply_strategy(build_context(3), "S2", budget=500)


if __name__ == "__main__":
    unittest.main()
