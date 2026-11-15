import random
import unittest
from itertools import product

from shift_scheduling import (DAY, NIGHT, OFF, PEOPLE, REQUESTS, penalty,
                              solve, violations)


def brute(people, days, requests, **kw):
    """모든 3^(사람×일) 배정을 검사기로 직접 확인하는 대조용 전수 탐색.

    반환: (최소 벌점, 하드 제약을 지키는 일정 수). 사람 3명 x 4일이면 3^12 = 531,441가지라 아주 작은 크기에서만 쓴다.
    """
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
        # 첫 가능 일정은 벌점 5, 노드 15(README). 최적이 아니라 처음 찾은 해라서 벌점이 더 높다
        self.assertEqual((r["penalty"], r["nodes"]), (5, 15))

    def test_optimal_schedule_is_valid_and_penalty_is_4(self):
        r = solve()
        self.assertEqual(violations(r["sched"]), [])
        # 벌점 4는 solve가 센 값과 penalty() 재계산값 둘 다 확인
        self.assertEqual(r["penalty"], 4)
        self.assertEqual(penalty(r["sched"]), 4)
        self.assertEqual(r["nodes"], 301)

    def test_pruning_does_not_change_optimum(self):
        # 가지치기를 꺼도 같은 최적 벌점 4: 가지치기는 속도만 바꾸고 답은 바꾸지 않는다(노드 421 대 301)
        full = solve(prune=False)
        self.assertEqual(full["penalty"], 4)
        self.assertEqual(full["solutions"], 48)   # 가능한 일정은 모두 48개
        self.assertEqual(full["nodes"], 421)
        self.assertLess(solve()["nodes"], full["nodes"])

    def test_night_is_followed_by_rest(self):
        # 야간 다음 날 휴무 규칙이 일정에서 실제로 지켜졌는지 검사기와 따로 확인
        s = solve()["sched"]
        for row in s.values():
            for a, b in zip(row, row[1:]):
                if a == NIGHT:
                    self.assertEqual(b, OFF)

    def test_tighter_rules_become_infeasible(self):
        # 필요한 근무 칸 = 7일 x (2+1) = 21 > 4명 x 5일 = 20이라 불가능. 야간도 7번 > 4명 x 1번
        self.assertIsNone(solve(max_work=5)["sched"])    # 21 근무일 > 4명 x 5일
        self.assertIsNone(solve(max_night=1)["sched"])   # 야간 7번 > 4명 x 1번
        self.assertIsNone(solve(max_consec=2)["sched"])

    def test_penalty_4_comes_from_consecutive_limit(self):
        # (아래 숫자는 README: 상한을 풀었을 때의 최소 벌점)
        # 벌점 하한은 3이고, 연속 근무 상한을 4일로 풀면 3까지 내려간다. 다른 상한을 풀어서는 4 그대로다.
        self.assertEqual(solve(max_consec=4)["penalty"], 3)
        self.assertEqual(solve(max_work=7)["penalty"], 4)
        self.assertEqual(solve(max_night=7)["penalty"], 4)

    def test_checker_catches_violations(self):
        # 검사기 자체의 시험: 전원 매일 낮 근무는 인원 수·연속 근무 위반, 야간 뒤 낮 근무는 휴식 위반
        bad = {p: [DAY] * 7 for p in PEOPLE}
        self.assertTrue(violations(bad))
        night_then_day = solve()["sched"]
        row = night_then_day[PEOPLE[0]]
        row[2], row[3] = NIGHT, DAY
        self.assertTrue(violations(night_then_day))

    def test_matches_brute_force_on_tiny_instances(self):
        # 고정 시드. 3명 x 3~4일의 무작위 휴무 희망으로 탐색 결과를 전수 탐색(brute)과 대조한다
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
