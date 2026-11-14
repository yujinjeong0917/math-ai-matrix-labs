import random
import unittest
from itertools import product

from shift_scheduling import (DAY, NIGHT, OFF, PEOPLE, REQUESTS, penalty,
                              solve, violations)


def brute(people, days, requests, **kw):
    """모든 3^(사람×일) 배정을 검사기로 직접 확인하는 대조용 전수 탐색."""
    best = None
    count = 0
    for cells in product((DAY, NIGHT, OFF), repeat=len(people) * days):
        sched = {p: list(cells[i * days:(i + 1) * days]) for i, p in enumerate(people)}
        if violations(sched, people=people, days=days, **kw):
            continue
        count += 1
        pen = penalty(sched, requests, days, people)
        if best is None or pen < best:
            best = pen
    return best, count


class ShiftSchedulingTest(unittest.TestCase):
    def test_first_feasible_schedule_is_valid(self):
        r = solve(optimize=False)
        self.assertEqual(violations(r["sched"]), [])
        self.assertEqual((r["penalty"], r["nodes"]), (5, 15))

    def test_optimal_schedule_is_valid_and_penalty_is_4(self):
        r = solve()
        self.assertEqual(violations(r["sched"]), [])
        self.assertEqual(r["penalty"], 4)
        self.assertEqual(penalty(r["sched"]), 4)
        self.assertEqual(r["nodes"], 301)

    def test_pruning_does_not_change_optimum(self):
        full = solve(prune=False)
        self.assertEqual(full["penalty"], 4)
        self.assertEqual(full["solutions"], 48)   # 가능한 일정은 모두 48개
        self.assertEqual(full["nodes"], 421)
        self.assertLess(solve()["nodes"], full["nodes"])

    def test_night_is_followed_by_rest(self):
        s = solve()["sched"]
        for row in s.values():
            for a, b in zip(row, row[1:]):
                if a == NIGHT:
                    self.assertEqual(b, OFF)

    def test_tighter_rules_become_infeasible(self):
        self.assertIsNone(solve(max_work=5)["sched"])    # 21 근무일 > 4명 x 5일
        self.assertIsNone(solve(max_night=1)["sched"])   # 야간 7번 > 4명 x 1번
        self.assertIsNone(solve(max_consec=2)["sched"])

    def test_penalty_4_comes_from_consecutive_limit(self):
        # 벌점 하한은 3이고, 연속 근무 상한을 4일로 풀면 3까지 내려간다. 다른 상한을 풀어서는 4 그대로다.
        self.assertEqual(solve(max_consec=4)["penalty"], 3)
        self.assertEqual(solve(max_work=7)["penalty"], 4)
        self.assertEqual(solve(max_night=7)["penalty"], 4)

    def test_checker_catches_violations(self):
        bad = {p: [DAY] * 7 for p in PEOPLE}
        self.assertTrue(violations(bad))
        night_then_day = solve()["sched"]
        row = night_then_day[PEOPLE[0]]
        row[2], row[3] = NIGHT, DAY
        self.assertTrue(violations(night_then_day))

    def test_matches_brute_force_on_tiny_instances(self):
        rng = random.Random(20261114)
        people = ("a", "b", "c")
        for days in (3, 3, 4):
            reqs = {(p, d) for p in people for d in range(days) if rng.random() < 0.5}
            kw = dict(need_day=1, need_night=1, max_consec=2, max_work=3, max_night=1)
            bb, n = brute(people, days, reqs, **kw)
            r = solve(people=people, days=days, requests=reqs, need_day=1, need_night=1,
                      max_consec=2, max_work=3, max_night=1)
            if n == 0:
                self.assertIsNone(r["sched"])
            else:
                self.assertEqual(r["penalty"], bb)
                self.assertEqual(r["solutions"] >= 1, True)


if __name__ == "__main__":
    unittest.main()
