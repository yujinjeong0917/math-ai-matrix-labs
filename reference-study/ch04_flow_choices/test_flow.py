import unittest
import flow as m


class Test(unittest.TestCase):
    def setUp(self):
        self.sc = m.load()

    def test_counts_add_up(self):
        for s in self.sc.values():
            c = m.counts(s)
            self.assertEqual(c["total"], c["same_page"] + c["external"] + c["internal"])
            self.assertEqual(c["total"], len(s["links"]))
            self.assertLessEqual(len(m.unique_addresses(s)), c["internal"])

    def test_counts_values(self):
        # 본문 숫자와 같은 값
        self.assertEqual(m.counts(self.sc["home"]), {"total": 67, "same_page": 0, "external": 0, "internal": 67, "internal_asset": 0})
        self.assertEqual(m.counts(self.sc["tracks"]), {"total": 225, "same_page": 19, "external": 1, "internal": 205, "internal_asset": 0})
        self.assertEqual(m.counts(self.sc["chapter"]), {"total": 30, "same_page": 0, "external": 2, "internal": 28, "internal_asset": 7})
        self.assertEqual([len(m.unique_addresses(self.sc[k])) for k in ("home", "tracks", "chapter")], [53, 193, 23])

    def test_anchor_links_are_same_page(self):
        for l in self.sc["tracks"]["links"]:
            if l["href"].startswith("#"):
                self.assertEqual(l["kind"], "same_page")

    def test_no_direct_link_home_to_chapter(self):
        self.assertEqual(m.direct_links_to_chapters(self.sc["home"]), [])
        self.assertEqual(self.sc["home"]["file_wide_chapter_hrefs"], [])  # 메뉴와 푸터까지 포함한 파일 전체 검색
        goal = self.sc["chapter"]["page"]
        self.assertNotIn(goal, m.graph(self.sc)[self.sc["home"]["page"]])
        self.assertIn(goal, m.graph(self.sc)[self.sc["tracks"]["page"]])

    def test_home_breakdown(self):
        self.assertEqual(m.target_breakdown(self.sc["home"]), {"tracks": 15, "concepts": 23, "matrix": 3, "other": 26})

    def test_chapter_choices_split(self):
        ch = self.sc["chapter"]
        self.assertEqual(len(m.choices(ch)), 21)
        self.assertEqual(len({l["resolved"] for l in m.choices(ch)}), 19)
        self.assertAlmostEqual(m.counts(self.sc["tracks"])["internal"] / len(m.choices(ch)), 9.76, places=2)

    def test_chapter_links_by_track(self):
        cb = m.chapter_links_by_track(self.sc["tracks"])
        self.assertEqual((sum(len(v) for v in cb.values()), len(cb), max(len(v) for v in cb.values())), (186, 17, 16))

    def test_steps(self):
        g = m.graph(self.sc)
        path = m.shortest_path(g, "/", self.sc["chapter"]["page"])
        self.assertEqual(path, ["/", "/tracks/", "/tracks/causal/06-difference-in-differences.html"])
        self.assertEqual(m.shortest_path(g, "/", "/"), ["/"])
        self.assertIsNone(m.shortest_path({"/a": set(), "/b": set()}, "/a", "/b"))

    def test_path_sums(self):
        h, t = self.sc["home"], self.sc["tracks"]
        self.assertEqual(m.counts(h)["internal"] + m.counts(t)["internal"], 272)
        a, b = m.unique_addresses(h), m.unique_addresses(t)
        self.assertEqual(len(a) + len(b), 246)
        self.assertEqual(len(a | b), 245)
        self.assertEqual(a & b, {"/work-ai/"})
        self.assertAlmostEqual(m.counts(t)["internal"] / m.counts(h)["internal"], 3.06, places=2)


if __name__ == "__main__":
    unittest.main()
