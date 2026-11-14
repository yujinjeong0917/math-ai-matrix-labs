import unittest
import interference as f


class Interference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = f.report()

    def test_graph_shape(self):
        self.assertAlmostEqual(self.r["mean_degree"], 11.7, delta=0.3)
        self.assertEqual(self.r["isolated"], 0)
        self.assertAlmostEqual(self.r["cross_edge_share"], 0.157, delta=0.01)

    def test_no_spillover_individual_ab_is_right(self):
        x = self.r["no_spillover"]
        self.assertAlmostEqual(x["truth"], 5.0, places=6)
        self.assertAlmostEqual(x["individual"], 5.0, delta=0.15)

    def test_positive_spillover_individual_ab_misses_it(self):
        x = self.r["spillover_pos"]
        self.assertAlmostEqual(x["truth"], 9.0, places=6)
        self.assertAlmostEqual(x["individual"], 5.0, delta=0.15)
        self.assertLess(abs(x["cluster"] - 9.0), abs(x["individual"] - 9.0))

    def test_negative_spillover_individual_ab_overstates(self):
        x = self.r["spillover_neg"]
        self.assertAlmostEqual(x["truth"], 1.0, places=6)
        self.assertAlmostEqual(x["individual"], 5.0, delta=0.15)
        self.assertLess(abs(x["cluster"] - 1.0), abs(x["individual"] - 1.0))

    def test_cluster_design_is_not_exact(self):
        x = self.r["spillover_pos"]
        self.assertGreater(9.0 - x["cluster"], 0.3)  # 커뮤니티 사이 간선 때문에 덜 맞는다


if __name__ == "__main__":
    unittest.main()
