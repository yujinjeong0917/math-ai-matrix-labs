import unittest

from calibration import CELLS, USERS, ai_accuracy, measures


class Calibration(unittest.TestCase):
    def test_cells(self):
        self.assertEqual(sum(CELLS.values()), 32)
        self.assertAlmostEqual(ai_accuracy(), 22 / 32)

    def test_p(self):
        m = measures(USERS["P"])
        self.assertAlmostEqual(m["reliance_high"], 15 / 16)
        self.assertAlmostEqual(m["reliance_low"], 7 / 16)
        self.assertAlmostEqual(m["awareness_gap"], 0.5)
        self.assertEqual(m["appropriate_count"], 22)
        self.assertAlmostEqual(m["overtrust"], 0.5)
        self.assertAlmostEqual(m["undertrust"], 5 / 22)

    def test_blind_has_no_gap_but_same_accuracy_as_p(self):
        q = measures(USERS["Q"])
        p = measures(USERS["P"])
        self.assertAlmostEqual(q["awareness_gap"], 0.0)
        self.assertEqual(q["appropriate_count"], p["appropriate_count"])  # 적정 신뢰만 보면 구별이 안 된다
        self.assertAlmostEqual(q["overtrust"], 1.0)
        self.assertAlmostEqual(q["undertrust"], 0.0)

    def test_r(self):
        m = measures(USERS["R"])
        self.assertAlmostEqual(m["reliance_high"], 0.5)
        self.assertAlmostEqual(m["reliance_low"], 0.5)
        self.assertAlmostEqual(m["awareness_gap"], 0.0)
        self.assertEqual(m["appropriate_count"], 16)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            measures({("높음", "맞음"): 15, ("높음", "틀림"): 0, ("낮음", "맞음"): 0, ("낮음", "틀림"): 0})


if __name__ == "__main__":
    unittest.main()
