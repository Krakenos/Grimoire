from unittest import TestCase

from grimoire.core.grimoire import filter_similar_entities


class TestFilterSimilarEntities(TestCase):
    def test_filter_with_no_relations(self):
        test_data = ["John", "Alex", "Jack"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual(test_results["John"], "John")
        self.assertEqual(test_results["Alex"], "Alex")
        self.assertEqual(test_results["Jack"], "Jack")

    def test_filter_with_relations(self):
        test_data = ["Alex", "Alexy", "Alexey"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual(test_results["Alex"], "Alexey")
        self.assertEqual(test_results["Alexy"], "Alexey")
        self.assertEqual(test_results["Alexey"], "Alexey")

    def test_filter_mixed(self):
        test_data = ["Alex", "Alexy", "Alexey", "John", "Jack", "Jerem", "Jeremy"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual(test_results["Alex"], "Alexey")
        self.assertEqual(test_results["Alexy"], "Alexey")
        self.assertEqual(test_results["Alexey"], "Alexey")
        self.assertEqual(test_results["John"], "John")
        self.assertEqual(test_results["Jack"], "Jack")
        self.assertEqual(test_results["Jerem"], "Jeremy")
        self.assertEqual(test_results["Jeremy"], "Jeremy")
