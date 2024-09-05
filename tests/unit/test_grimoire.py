from unittest import TestCase

from grimoire.core.grimoire import filter_similar_entities


class TestFilterSimilarEntities(TestCase):
    def test_filter_with_no_relations(self):
        test_data = ["John", "Alex", "Jack"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual("John", test_results["John"])
        self.assertEqual("Alex", test_results["Alex"])
        self.assertEqual("Jack", test_results["Jack"])

    def test_filter_with_relations(self):
        test_data = ["Alex", "Alexy", "Alexey"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual(
            "Alex",
            test_results["Alex"],
        )
        self.assertEqual(
            "Alex",
            test_results["Alexy"],
        )
        self.assertEqual("Alex", test_results["Alexey"])

    def test_filter_mixed(self):
        test_data = ["Alex", "Alexy", "Alexey", "John", "Jack", "Jerem", "Jeremy", "Jeremey"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual("Alex", test_results["Alex"])
        self.assertEqual("Alex", test_results["Alexy"])
        self.assertEqual("Alex", test_results["Alexey"])
        self.assertEqual("John", test_results["John"])
        self.assertEqual("Jack", test_results["Jack"])
        self.assertEqual("Jerem", test_results["Jerem"])
        self.assertEqual("Jerem", test_results["Jeremy"])
        self.assertEqual("Jerem", test_results["Jeremey"])

    def test_filter_uppercase_priority(self):
        test_data = ["Alex", "alex"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual("Alex", test_results["Alex"])
        self.assertEqual("Alex", test_results["alex"])

    def test_filter_length_priority(self):
        test_data = ["Alex", "Alexey"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual("Alex", test_results["Alex"])
        self.assertEqual("Alex", test_results["Alexey"])

    def test_filter_with_relations_uppercase_priority(self):
        test_data = ["Alex", "Alexy", "Alexey", "ALEX"]
        test_results = filter_similar_entities(test_data)
        self.assertEqual("ALEX", test_results["Alex"])
        self.assertEqual("ALEX", test_results["Alexy"])
        self.assertEqual("ALEX", test_results["Alexey"])
        self.assertEqual("ALEX", test_results["ALEX"])
