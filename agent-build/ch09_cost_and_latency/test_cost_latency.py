import unittest

import cost_latency as c


class CostLatencyTest(unittest.TestCase):
    def test_input_resend_by_hand(self):
        # 손계산: 호출마다 시스템 800 + 쌓인 이력
        # 호출 1~5의 입력 = 시스템 800 + 쌓인 이력(60, 250, 410, 850, 950). 이력은 직전 출력 40과 도구 결과(150, 120, 400, 60)가 더해진 값이다.
        by_hand = [800 + 60, 800 + 250, 800 + 410, 800 + 850, 800 + 950]
        self.assertEqual(sum(by_hand), 6520)
        self.assertEqual(c.run_flow(c.serial_flow())['input_tokens'], 6520)

    def test_serial_vs_parallel(self):
        s, p = c.run_flow(c.serial_flow()), c.run_flow(c.parallel_flow())
        self.assertEqual((s['calls'], p['calls']), (5, 3))
        self.assertEqual((s['input_tokens'], p['input_tokens']), (6520, 4260))
        # README 표의 13.008초와 10.604초. 출력 토큰은 둘이 같아(280) 차이는 호출 수와 입력 재전송에서 나온다.
        self.assertEqual((s['seconds'], p['seconds']), (13.008, 10.604))
        self.assertEqual(s['output_tokens'], p['output_tokens'])

    def test_cost_is_formula_not_price(self):
        s = c.run_flow(c.serial_flow())
        # 출력 280토큰이라 6520 + 4 x 280. 단가는 임의 값이라 비용 식 자체만 확인한다.
        self.assertEqual(c.cost_units(s, 1, 4), 6520 + 4 * 280)
        self.assertEqual(c.cost_units(s, 0, 0), 0)

    def test_approval_wait_adds_linearly(self):
        base = c.run_flow(c.serial_flow())['seconds']
        waited = c.run_flow(c.serial_flow(), {**c.PARAMS, 'approval_wait_s': 120.0})['seconds']
        # 부동소수점 합이라 소수 셋째 자리까지만 비교(정확히 같은지 비교하지 않는다).
        self.assertAlmostEqual(waited - base, 120.0, places=3)

    def test_endless_loop_grows_faster_than_calls(self):
        r = c.report()['endless_error_loop']
        self.assertEqual([r[k]['calls'] for k in (10, 20, 50)], [10, 20, 50])
        # README: 정상 6520 대비 2.51배, 6.45배, 25.64배. 끝까지 오류라 done은 모두 False.
        self.assertEqual([r[k]['input_tokens'] for k in (10, 20, 50)], [16350, 42050, 167150])
        self.assertEqual([r[k]['input_x_normal'] for k in (10, 20, 50)], [2.51, 6.45, 25.64])
        self.assertTrue(all(not r[k]['done'] for k in r))

    def test_cap_sweep(self):
        # 단계 상한별로 안내까지 못 간 문의 수와 그중 환불은 나간 수. 상한 4는 정상 문의(5호출)도 모두 멈춰 20건.
        got = c.report()['stopped_by_cap']
        self.assertEqual(got[4], {'stopped': 20, 'refunded_but_not_told': 12})
        self.assertEqual(got[5], {'stopped': 8, 'refunded_but_not_told': 4})
        self.assertEqual(got[8], {'stopped': 2, 'refunded_but_not_told': 0})
        self.assertEqual(got[20], {'stopped': 1, 'refunded_but_not_told': 0})
        self.assertEqual(len(c.FAILURES), 20)

    def test_deterministic(self):
        self.assertEqual(c.report(), c.report())


if __name__ == '__main__':
    unittest.main()
