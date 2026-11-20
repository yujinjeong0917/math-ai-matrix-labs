import unittest

from reliance import AI_CORRECT, USERS, metrics


class Reliance(unittest.TestCase):
    def test_ai_accuracy(self):
        self.assertEqual(sum(AI_CORRECT), 7)

    def test_blind_follower(self):
        m = metrics(AI_CORRECT, USERS["A"])
        self.assertAlmostEqual(m["reliance"], 1.0)
        self.assertAlmostEqual(m["appropriate"], 0.7)  # AI 정확도와 같다
        self.assertAlmostEqual(m["overtrust"], 1.0)
        self.assertAlmostEqual(m["undertrust"], 0.0)

    def test_selective_user(self):
        m = metrics(AI_CORRECT, USERS["B"])
        self.assertAlmostEqual(m["reliance"], 0.7)
        self.assertAlmostEqual(m["appropriate"], 0.8)
        self.assertAlmostEqual(m["overtrust"], 1 / 3)
        self.assertAlmostEqual(m["undertrust"], 1 / 7)

    def test_refuser(self):
        m = metrics(AI_CORRECT, USERS["C"])
        self.assertAlmostEqual(m["reliance"], 0.0)
        self.assertAlmostEqual(m["appropriate"], 0.3)
        self.assertAlmostEqual(m["undertrust"], 1.0)

    def test_counts_sum(self):
        for f in USERS.values():
            self.assertEqual(sum(metrics(AI_CORRECT, f)["counts"].values()), 10)

    def test_length_mismatch(self):
        with self.assertRaises(ValueError):
            metrics([True], [True, False])


if __name__ == "__main__":
    unittest.main()
