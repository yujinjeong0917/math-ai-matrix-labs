import unittest
import simulation as sim

SEEDS = list(range(1, 21))


class SimulationTest(unittest.TestCase):
    def test_engine_matches_mm1_theory(self):
        got, theory = sim.mm1_check()
        self.assertAlmostEqual(theory, 20.0)
        self.assertLess(abs(got - theory) / theory, 0.03)

    def test_same_seed_same_result(self):
        self.assertEqual(sim.one_replication(8, 2), sim.one_replication(8, 2))
        self.assertNotEqual(sim.one_replication(8, 2), sim.one_replication(9, 2))

    def test_single_runs_differ_a_lot(self):
        xs = sim.replicate(2, SEEDS)
        self.assertAlmostEqual(min(xs), 0.62, places=2)
        self.assertAlmostEqual(max(xs), 13.98, places=2)
        self.assertGreater(max(xs) / min(xs), 10)

    def test_confidence_interval(self):
        m, lo, hi = sim.ci95(sim.replicate(2, SEEDS))
        self.assertAlmostEqual(m, 3.77, places=2)
        self.assertAlmostEqual(lo, 2.20, places=2)
        self.assertAlmostEqual(hi, 5.34, places=2)

    def test_extra_agent_cuts_wait(self):
        m3, lo3, hi3 = sim.ci95(sim.replicate(3, SEEDS))
        self.assertAlmostEqual(m3, 0.58, places=2)
        self.assertLess(hi3, 2.20)             # 두 구간이 겹치지 않는다

    def test_common_random_numbers(self):
        d = [a - b for a, b in zip(sim.replicate(2, SEEDS), sim.replicate(3, SEEDS))]
        m, lo, hi = sim.ci95(d)
        self.assertAlmostEqual(m, 3.19, places=2)
        d2 = [a - b for a, b in zip(sim.replicate(2, SEEDS), sim.replicate(3, range(101, 121)))]
        m2, lo2, hi2 = sim.ci95(d2)
        self.assertLess(hi - lo, hi2 - lo2)    # 같은 난수로 짝지으면 구간이 좁다

    def test_peak_hours_hidden_by_daily_mean(self):
        pk = []
        for s in SEEDS:
            arr = sim.make_arrivals(s)
            w = sim.run(arr, sim.make_services(s, len(arr)), 2)
            pk.append(sum(x for a, x in zip(arr, w) if a < sim.PEAK_END) / sum(1 for a in arr if a < sim.PEAK_END))
        self.assertGreater(sum(pk) / 20, sim.ci95(sim.replicate(2, SEEDS))[0])


if __name__ == "__main__":
    unittest.main()
