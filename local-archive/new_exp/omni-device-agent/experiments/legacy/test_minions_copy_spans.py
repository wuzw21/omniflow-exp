import unittest
from analyze_minions_copy_spans import spans


class CopySpanTests(unittest.TestCase):
    def test_verbatim_copy_includes_offsets_without_counting_short_overlap(self):
        source = list(range(100))
        output = [1000] + source[20:50] + [1001] + source[70:75]
        self.assertEqual(spans(source, output), [{'source_start': 20, 'output_start': 1, 'tokens': 30}])

    def test_repeated_tokens_do_not_create_an_impossible_copy_count(self):
        source = [1] * 100
        output = [1] * 50
        result = spans(source, output)
        self.assertEqual(sum(m['tokens'] for m in result), 50)
        self.assertEqual(spans(source, [2] * 50), [])


if __name__ == '__main__':
    unittest.main()
