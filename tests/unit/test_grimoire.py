from unittest import TestCase

from grimoire.core.grimoire import filter_similar_entities


class TestFilterSimilarEntities(TestCase):
    def test_filter_with_no_relations(self):
        test_data = ["John", "Alex", "Jack"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual(test_results["John"], "John")
        self.assertEqual(test_results["Alex"], "Alex")
        self.assertEqual(test_results["Jack"], "Jack")
