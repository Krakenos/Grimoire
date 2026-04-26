from unittest import TestCase

from grimoire.core.grimoire import filter_similar_entities

PERSON = "PERSON"
GPE = "GPE"
ORG = "ORG"


def labels(*names: str, label: str = PERSON) -> dict[str, str]:
    return dict.fromkeys(names, label)


class TestFilterSimilarEntities(TestCase):
    def test_filter_with_no_relations(self):
        test_data = ["John", "Alex", "Jack"]
        test_results = filter_similar_entities(test_data, labels("John", "Alex", "Jack"))
        self.assertEqual("John", test_results["John"])
        self.assertEqual("Alex", test_results["Alex"])
        self.assertEqual("Jack", test_results["Jack"])

    def test_filter_with_relations(self):
        test_data = ["Alex", "Alexy", "Alexey"]
        test_results = filter_similar_entities(test_data, labels("Alex", "Alexy", "Alexey"))
        self.assertEqual("Alex", test_results["Alex"])
        self.assertEqual("Alex", test_results["Alexy"])
        self.assertEqual("Alex", test_results["Alexey"])

    def test_filter_mixed(self):
        test_data = ["Alex", "Alexy", "Alexey", "John", "Jack", "Jerem", "Jeremy", "Jeremey"]
        test_results = filter_similar_entities(test_data, labels(*test_data))
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
        test_results = filter_similar_entities(test_data, labels("Alex", "alex"))
        self.assertEqual("Alex", test_results["Alex"])
        self.assertEqual("Alex", test_results["alex"])

    def test_filter_length_priority(self):
        test_data = ["Alex", "Alexey"]
        test_results = filter_similar_entities(test_data, labels("Alex", "Alexey"))
        self.assertEqual("Alex", test_results["Alex"])
        self.assertEqual("Alex", test_results["Alexey"])

    def test_filter_with_relations_uppercase_priority(self):
        test_data = ["Alex", "Alexy", "Alexey", "ALEX"]
        test_results = filter_similar_entities(test_data, labels("Alex", "Alexy", "Alexey", "ALEX"))
        self.assertEqual("ALEX", test_results["Alex"])
        self.assertEqual("ALEX", test_results["Alexy"])
        self.assertEqual("ALEX", test_results["Alexey"])
        self.assertEqual("ALEX", test_results["ALEX"])

    def test_filter_transitive_grouping(self):
        # A~B, B~C, but A does not directly meet threshold with C alone
        # All three must converge to same canonical via union-find
        test_data = ["Jon", "John", "Johnny"]
        test_results = filter_similar_entities(test_data, labels("Jon", "John", "Johnny"))
        canonical = test_results["Jon"]
        self.assertEqual(canonical, test_results["John"])
        self.assertEqual(canonical, test_results["Johnny"])

    def test_filter_no_substring_merging(self):
        # "Lin" must not merge with "Lincoln" — ratio-based scorer prevents substring inflation
        test_data = ["Lin", "Lincoln"]
        test_results = filter_similar_entities(test_data, labels("Lin", "Lincoln"))
        self.assertNotEqual(test_results["Lin"], test_results["Lincoln"])

    def test_filter_cross_label_no_merge(self):
        # Same-ish spelling but different spaCy labels must not merge
        test_data = ["Banks", "Bank"]
        entity_labels = {"Banks": PERSON, "Bank": ORG}
        test_results = filter_similar_entities(test_data, entity_labels)
        self.assertNotEqual(test_results["Banks"], test_results["Bank"])

    def test_filter_same_label_merges(self):
        # Confirm label guard does not block valid same-label merges
        test_data = ["Banks", "Bank"]
        test_results = filter_similar_entities(test_data, labels("Banks", "Bank"))
        self.assertEqual(test_results["Banks"], test_results["Bank"])

    def test_filter_empty_input(self):
        self.assertEqual({}, filter_similar_entities([], {}))
