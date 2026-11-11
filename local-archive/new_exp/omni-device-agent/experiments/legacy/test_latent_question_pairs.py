import unittest
from types import SimpleNamespace

from latent_question_pairs import select_pairs


def row(i, question, answer, source="source", receiver="receiver"):
    return SimpleNamespace(example_id=str(i), question=question, answers=(answer,),
                           sharer_context=source, receiver_context=receiver)


class QuestionPairTests(unittest.TestCase):
    def test_same_answer_rephrase_is_not_a_changed_answer_control(self):
        a = row(0, "Who?", "A")
        b = row(1, "Who exactly?", "a")
        c = row(2, "Who else?", "B")
        self.assertEqual(select_pairs([a, b, c], 1), [(a, c)])

    def test_swapped_evidence_assignment_is_not_same_context(self):
        a = row(0, "Who?", "A")
        b = row(1, "Who else?", "B", "receiver", "source")
        with self.assertRaises(ValueError):
            select_pairs([a, b], 1)

    def test_validation_order_is_deterministic_and_groups_are_distinct(self):
        rows = [row(0, "Q0", "A"), row(1, "Q1", "B"), row(2, "Q2", "C"),
                row(3, "Q3", "A", "S2"), row(4, "Q4", "B", "S2")]
        self.assertEqual([(a.example_id, b.example_id) for a, b in select_pairs(rows, 2)],
                         [("0", "1"), ("3", "4")])


if __name__ == "__main__":
    unittest.main()
