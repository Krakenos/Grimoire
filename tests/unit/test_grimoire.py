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
        test_data = ["Alex", "Alexy", "Alexey", "John", "Jack", "Jerem", "Jeremy", "Jeremey"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual(test_results["Alex"], "Alexey")
        self.assertEqual(test_results["Alexy"], "Alexey")
        self.assertEqual(test_results["Alexey"], "Alexey")
        self.assertEqual(test_results["John"], "John")
        self.assertEqual(test_results["Jack"], "Jack")
        self.assertEqual(test_results["Jerem"], "Jeremy")
        self.assertEqual(test_results["Jeremy"], "Jeremy")
        self.assertEqual(test_results["Jeremey"], "Jeremy")

    def test_filter_uppercase_priority(self):
        test_data = ["Alex", "alex"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual(test_results["Alex"], "Alex")
        self.assertEqual(test_results["alex"], "Alex")

    def test_filter_length_priority(self):
        test_data = ["Alex", "Alexey"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual(test_results["Alex"], "Alex")
        self.assertEqual(test_results["Alexey"], "Alex")

    def test_filter_with_relations_uppercase_priority(self):
        test_data = ["Alex", "Alexy", "Alexey", "ALEXEY"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual(test_results["Alex"], "ALEXEY")
        self.assertEqual(test_results["Alexy"], "ALEXEY")
        self.assertEqual(test_results["Alexey"], "ALEXEY")
        self.assertEqual(test_results["ALEXEY"], "ALEXEY")
