"""Generated decision sources: smart-home handling and causal-ladder questions."""
import random
import unittest

from decision.data import synth_causal as C
from decision.data import synth_home as H


class HomeTests(unittest.TestCase):
    def home(self):
        return {"policy": {"multi_action": "atomic", "allow_partial": False, "security_mode": "home"},
                "devices": [{"id": "a", "name": "Kettle", "room": "kitchen", "kind": "appliance", "available": True, "power": False},
                            {"id": "b", "name": "Fan", "room": "den", "kind": "appliance", "available": False, "power": False},
                            {"id": "l", "name": "Lamp", "room": "den", "kind": "light", "available": True, "power": True, "brightness": 40}],
                "schedules": [], "rules": []}

    def test_atomic_request_with_an_offline_device_is_blocked(self):
        result = H.resolve(self.home(), {"kind": "execute", "ops": [["a", "power", True], ["b", "power", True]]})
        self.assertEqual((result["status"], result["changed"]), ("blocked", []))

    def test_best_effort_runs_the_available_part(self):
        home = self.home()
        home["policy"]["allow_partial"] = True
        result = H.resolve(home, {"kind": "execute_partial", "ops": [["a", "power", True], ["b", "power", True]]})
        self.assertEqual((result["status"], result["changed"]), ("execute_partial", [["a", "power", True]]))

    def test_ordered_steps_count_only_real_changes(self):
        result = H.resolve(self.home(), {"kind": "execute", "ops": [["l", "power", True], ["l", "power", False]]})
        self.assertEqual(result["changed"], [["l", "power", False]])
        self.assertEqual(H.resolve(self.home(), {"kind": "execute", "ops": [["l", "power", True]]})["status"], "no_change")

    def test_clarify_has_no_justified_targets(self):
        home = self.home()
        result = H.resolve(home, {"kind": "clarify", "candidates": ["a", "l"]})
        questions = H.questions_for(home, {"kind": "clarify", "candidates": ["a", "l"]}, result, random.Random(0))
        targets = [q for q in questions if q[0].startswith("target:")]
        self.assertTrue(all(q[3] == H.NO for q in targets))
        self.assertEqual(questions[0][3], "ask a clarifying question")

    def test_requests_are_well_formed(self):
        for state, questions, meta in H.requests(0, 200):
            for role, _, values, gold in questions:
                self.assertIn(gold, values, role)
                self.assertEqual(len(values), len(set(values)), role)


class CausalTests(unittest.TestCase):
    def test_backdoor_adjustment_matches_the_intervention(self):
        model = C.Model("confounding", random.Random(1))
        ate = model.prob({"Y": 1}, do={"X": 1}) - model.prob({"Y": 1}, do={"X": 0})
        adjusted = sum(model.prob({"V1": v}) * (model.prob({"Y": 1}, {"V1": v, "X": 1}) - model.prob({"Y": 1}, {"V1": v, "X": 0}))
                       for v in (0, 1))
        self.assertAlmostEqual(ate, adjusted)

    def test_labels_are_balanced(self):
        golds = [gold for _, gold, _, _ in C.questions(0, 1000)]
        self.assertGreater(golds.count("yes") / len(golds), .4)
        self.assertLess(golds.count("yes") / len(golds), .6)


if __name__ == "__main__":
    unittest.main()
