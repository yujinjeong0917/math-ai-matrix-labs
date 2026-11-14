import random
import unittest

from max_flow import (EDGES, NODES, TIE_EDGES, all_cuts, cut_edges,
                      edmonds_karp, min_cut_from_residual, min_cuts,
                      upgrade_table)


class MaxFlowTest(unittest.TestCase):
    def test_max_flow_value_and_paths(self):
        value, flow, paths, _ = edmonds_karp(EDGES)
        self.assertEqual(value, 11)
        self.assertEqual(sum(p for _, p in paths), 11)
        self.assertEqual(len(paths), 4)

    def test_flow_conservation_and_capacity(self):
        _, flow, _, _ = edmonds_karp(EDGES)
        for e, f in flow.items():
            self.assertTrue(0 <= f <= EDGES[e])
        for n in ("A", "B", "C", "D"):
            inn = sum(f for (u, v), f in flow.items() if v == n)
            out = sum(f for (u, v), f in flow.items() if u == n)
            self.assertEqual(inn, out)

    def test_max_flow_equals_min_cut_by_enumeration(self):
        value = edmonds_karp(EDGES)[0]
        cuts = all_cuts(EDGES)
        self.assertEqual(len(cuts), 16)
        self.assertEqual(cuts[0][0], value)

    def test_residual_cut_is_a_min_cut(self):
        value, _, _, res = edmonds_karp(EDGES)
        _, cut = min_cut_from_residual(EDGES, res)
        self.assertEqual(sorted(cut), [("C", "T"), ("D", "T")])
        self.assertEqual(sum(EDGES[e] for e in cut), value)

    def test_upgrade_table(self):
        base, rows = upgrade_table(EDGES, 2)
        gain = {e: d for e, _, d in rows}
        self.assertEqual(base, 11)
        self.assertEqual(gain[("C", "T")], 1)   # 2 늘렸지만 1만 늘어난다
        self.assertEqual(gain[("D", "T")], 1)
        self.assertEqual(sum(v for k, v in gain.items() if k not in (("C", "T"), ("D", "T"))), 0)

    def test_tie_graph_two_min_cuts(self):
        self.assertEqual(edmonds_karp(TIE_EDGES)[0], 12)
        self.assertEqual(len(min_cuts(TIE_EDGES)), 2)
        _, rows = upgrade_table(TIE_EDGES, 2)
        gain = {e: d for e, _, d in rows}
        self.assertEqual(gain[("D", "T")], 0)   # 두 최소절단 중 하나에만 있는 간선
        self.assertEqual(gain[("A", "D")], 0)
        self.assertEqual(gain[("C", "T")], 1)   # 둘 모두에 있어도 세 번째 절단(13)이 막는다

    def test_random_graphs_flow_equals_min_cut(self):
        rng = random.Random(20261114)
        for _ in range(100):
            edges = {}
            for u, v in [("S", "A"), ("S", "B"), ("A", "B"), ("B", "A"), ("A", "T"), ("B", "T"), ("A", "C"), ("C", "T"), ("S", "C")]:
                if rng.random() < 0.8:
                    edges[(u, v)] = rng.randint(1, 9)
            nodes = ("S", "A", "B", "C", "T")
            self.assertEqual(edmonds_karp(edges)[0], all_cuts(edges, nodes)[0][0])


if __name__ == "__main__":
    unittest.main()
