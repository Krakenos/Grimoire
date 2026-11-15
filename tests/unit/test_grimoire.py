from types import SimpleNamespace
from unittest import TestCase, expectedFailure

from grimoire.core.entity_filters import (
    _clean_entity_text,
    _discriminating_tokens,
    _is_acceptable_entity,
    _passes_threshold,
    filter_similar_entities,
)
from grimoire.core.grimoire import _normalize_text

PERSON = "PERSON"
GPE = "GPE"
ORG = "ORG"
NORP = "NORP"
WORK_OF_ART = "WORK_OF_ART"


def labels(*names: str, label: str = PERSON) -> dict[str, str]:
    return dict.fromkeys(names, label)


class FakeSpan:
    """Minimal spaCy-Span-shaped object for `_is_acceptable_entity` tests.

    `pos_tags` defaults to PROPN per whitespace-token, which is the typical
    case for proper nouns. Pass explicit tags to test POS-gate behaviour.
    """

    def __init__(self, text: str, pos_tags: list[str] | None = None):
        self.text = text
        words = text.split()
        if pos_tags is None:
            pos_tags = ["PROPN"] * len(words)
        self._tokens = [SimpleNamespace(pos_=p) for p in pos_tags]

    def __iter__(self):
        return iter(self._tokens)


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

    # --- Canonical selection ---

    def test_canonical_prefers_more_frequent_form(self):
        # "Claire" (frequent, correct) must win over "Clare" (rare typo) even though
        # the typo is shorter — frequency dominates the canonical tie-break.
        test_data = ["Clare", "Claire"]
        name_counts = {"Clare": 2, "Claire": 40}
        test_results = filter_similar_entities(test_data, labels(*test_data), name_counts)
        self.assertEqual("Claire", test_results["Clare"])
        self.assertEqual("Claire", test_results["Claire"])

    def test_canonical_frequency_overrides_length(self):
        # The longer form wins when it is the dominant spelling.
        test_data = ["Alex", "Alexey"]
        name_counts = {"Alex": 1, "Alexey": 25}
        test_results = filter_similar_entities(test_data, labels(*test_data), name_counts)
        self.assertEqual("Alexey", test_results["Alex"])
        self.assertEqual("Alexey", test_results["Alexey"])

    def test_canonical_falls_back_to_length_on_equal_counts(self):
        # With equal counts the tie-break falls back to shortest-first, preserving
        # the previous behaviour.
        test_data = ["Alex", "Alexey"]
        test_results = filter_similar_entities(test_data, labels(*test_data), {"Alex": 5, "Alexey": 5})
        self.assertEqual("Alex", test_results["Alex"])
        self.assertEqual("Alex", test_results["Alexey"])

    def test_canonical_without_counts_unchanged(self):
        # Omitting name_counts must not change the legacy length/lexical behaviour.
        test_data = ["Alex", "Alexey"]
        test_results = filter_similar_entities(test_data, labels(*test_data))
        self.assertEqual("Alex", test_results["Alex"])
        self.assertEqual("Alex", test_results["Alexey"])

    # --- New gates ---

    def test_filter_short_lowercase_no_merge(self):
        # "as"/"was" and "Le"/"let" must not merge — short-string strict threshold
        test_data = ["as", "was"]
        test_results = filter_similar_entities(test_data, labels(*test_data))
        self.assertNotEqual(test_results["as"], test_results["was"])

    def test_filter_short_lowercase_no_merge_le_let(self):
        test_data = ["Le", "let"]
        test_results = filter_similar_entities(test_data, labels(*test_data))
        self.assertNotEqual(test_results["Le"], test_results["let"])

    def test_filter_short_capitalized_merges(self):
        # Short proper nouns (both capitalised) must still merge at normal threshold
        # Jon/John scores ≈ 86 which passes 80 but would fail 95
        test_data = ["Jon", "John"]
        test_results = filter_similar_entities(test_data, labels(*test_data))
        self.assertEqual(test_results["Jon"], test_results["John"])

    def test_filter_discriminating_token_blocks_empire_merge(self):
        # "the Nur Empire" and "the Loro Empire" share determiners/suffix but
        # have different discriminating tokens (Nur vs Loro) and must not merge.
        test_data = ["the Nur Empire", "the Loro Empire"]
        test_results = filter_similar_entities(test_data, labels(*test_data, label=GPE))
        self.assertNotEqual(test_results["the Nur Empire"], test_results["the Loro Empire"])

    def test_filter_discriminating_token_allows_case_variants(self):
        # "the Royal Academy" and "Royal Academy" differ only in the dropped
        # determiner — discriminating tokens are identical so they must merge.
        test_data = ["the Royal Academy", "Royal Academy"]
        test_results = filter_similar_entities(test_data, labels(*test_data, label=ORG))
        self.assertEqual(test_results["the Royal Academy"], test_results["Royal Academy"])


class TestCleanEntityText(TestCase):
    def test_strips_possessive_apostrophe_s(self):
        self.assertEqual("Cooley III", _clean_entity_text("Cooley III's"))

    def test_strips_unicode_possessive(self):
        self.assertEqual("Claire François", _clean_entity_text("Claire François’s"))

    def test_collapses_double_spaces(self):
        self.assertEqual("the Bauer Kingdom", _clean_entity_text("the  Bauer Kingdom"))

    def test_strips_leading_trailing_punctuation(self):
        self.assertEqual("Yu", _clean_entity_text('"Yu"'))

    def test_normalises_nbsp(self):
        # Non-breaking space (U+00A0) should collapse to a regular space
        self.assertEqual("the Royal Academy", _clean_entity_text("the Royal Academy"))

    def test_empty_after_strip(self):
        self.assertEqual("", _clean_entity_text("  '  "))


class TestNormalizeText(TestCase):
    def test_collapses_intraline_spaces(self):
        self.assertEqual("hello world", _normalize_text("hello  world"))

    def test_collapses_excess_blank_lines(self):
        result = _normalize_text("a\n\n\n\nb")
        self.assertEqual("a\n\nb", result)

    def test_strips_leading_trailing(self):
        self.assertEqual("hello", _normalize_text("  hello  "))

    def test_nfkc_normalisation(self):
        # Fullwidth letter → ASCII
        self.assertEqual("A", _normalize_text("Ａ"))


class TestPassesThreshold(TestCase):
    def test_blocks_short_lowercase_pair(self):
        # Both sides short, one lowercase — strict threshold applies
        self.assertFalse(_passes_threshold("as", "was", 80.0))

    def test_blocks_when_one_side_lowercase(self):
        # "Le" (upper) + "let" (lower) — one side lowercase triggers strict gate
        self.assertFalse(_passes_threshold("Le", "let", 80.0))

    def test_allows_short_capitalised_pair(self):
        # Both sides capitalised — skip strict gate, use normal threshold
        self.assertTrue(_passes_threshold("Jon", "John", 86.0))

    def test_allows_longer_names(self):
        # min_len > 3 — always uses normal threshold
        self.assertTrue(_passes_threshold("Alex", "Alexy", 89.0))
        self.assertTrue(_passes_threshold("Banks", "Bank", 89.0))

    def test_blocks_score_below_normal(self):
        self.assertFalse(_passes_threshold("Alpha", "Omega", 50.0))


class TestDiscriminatingTokens(TestCase):
    def test_drops_stopwords(self):
        self.assertEqual({"nur", "empire"}, _discriminating_tokens("the Nur Empire"))

    def test_single_word(self):
        self.assertEqual({"claire"}, _discriminating_tokens("Claire"))

    def test_all_stopwords(self):
        self.assertEqual(set(), _discriminating_tokens("the a an"))

    def test_multi_word_no_stopwords(self):
        self.assertEqual({"royal", "academy"}, _discriminating_tokens("Royal Academy"))


class TestIsAcceptableEntity(TestCase):
    def test_accepts_proper_noun(self):
        self.assertTrue(_is_acceptable_entity(FakeSpan("Claire"), PERSON))

    def test_accepts_multi_word(self):
        self.assertTrue(_is_acceptable_entity(FakeSpan("the Royal Academy"), ORG))

    def test_rejects_banned_label(self):
        self.assertFalse(_is_acceptable_entity(FakeSpan("2024"), "DATE"))

    def test_rejects_trivial_text(self):
        self.assertFalse(_is_acceptable_entity(FakeSpan("and", ["CCONJ"]), PERSON))

    def test_rejects_function_word_pos(self):
        # Span whose every token is DET/ADP/etc. is rejected regardless of label
        self.assertFalse(_is_acceptable_entity(FakeSpan("the", ["DET"]), ORG))
        self.assertFalse(_is_acceptable_entity(FakeSpan("by", ["ADP"]), PERSON))

    def test_rejects_short_lowercase_noun(self):
        # Single-token, all-lowercase, ≤6 chars → common-noun FP
        self.assertFalse(_is_acceptable_entity(FakeSpan("host", ["NOUN"]), ORG))
        self.assertFalse(_is_acceptable_entity(FakeSpan("noble", ["NOUN"]), ORG))

    def test_rejects_work_of_art_single_token(self):
        self.assertFalse(_is_acceptable_entity(FakeSpan("ran", ["VERB"]), WORK_OF_ART))

    def test_rejects_work_of_art_lowercase_multiword(self):
        self.assertFalse(_is_acceptable_entity(FakeSpan("some title", ["ADJ", "NOUN"]), WORK_OF_ART))

    def test_accepts_work_of_art_capitalised_multiword(self):
        self.assertTrue(_is_acceptable_entity(FakeSpan("The Hateful Cry", ["DET", "ADJ", "NOUN"]), WORK_OF_ART))

    def test_rejects_too_short(self):
        self.assertFalse(_is_acceptable_entity(FakeSpan("A", ["PROPN"]), PERSON))

    def test_possessive_stripped_before_length_check(self):
        # "Yu's" → cleaned to "Yu" (len 2) → accepted
        self.assertTrue(_is_acceptable_entity(FakeSpan("Yu's", ["PROPN"]), PERSON))

    def test_accepts_adjectival_demonyms(self):
        # GUARDRAIL: legitimate NORP demonyms are POS-tagged ADJ (not PROPN/NOUN).
        # Any "require a nominal token" or "drop single-token ADJ" filter would
        # delete these valid entities — see docs/entity_disambiguation.md §D.
        for demonym in ("Japanese", "English", "Greek", "Buddhist"):
            with self.subTest(demonym=demonym):
                self.assertTrue(_is_acceptable_entity(FakeSpan(demonym, ["ADJ"]), NORP))

    def test_rejects_trivial_word_list_entries(self):
        # The hand-curated TRIVIAL_ENTITY_TEXTS set catches content-POS verbs,
        # adjectives and capitalised-at-sentence-start common words that the POS
        # and shape gates miss. Lock the documented entries in as regressions.
        cases = {
            "grinned": "VERB",
            "laughed": "VERB",
            "yawned": "VERB",
            "shamed": "VERB",
            "ran": "VERB",
            "shrill": "ADJ",
            "sure": "ADJ",
            "danger": "NOUN",
            "without": "ADP",
        }
        for text, pos in cases.items():
            with self.subTest(text=text):
                self.assertFalse(_is_acceptable_entity(FakeSpan(text, [pos]), PERSON))


class TestKnownResiduals(TestCase):
    """Documented limitations of the string/POS tier (docs/entity_disambiguation.md).

    These assert the *desired* behaviour and are marked expectedFailure: they
    cannot be fixed without semantic signal (coreference / LLM) and would cause
    unacceptable collateral if forced. They exist so the residuals stay visible
    and so a future semantic tier has ready-made acceptance tests — if one starts
    passing, drop the decorator.
    """

    @expectedFailure
    def test_single_substitution_over_merge(self):
        # §A: "Marcel"/"Darcel" — ratio 83.3, both 6 chars and capitalised, so the
        # short-string gate does not apply. No safe string threshold separates this
        # from legitimate one-edit pairs (Clare/Claire = 90.9) in the same document.
        results = filter_similar_entities(["Marcel", "Darcel"], labels("Marcel", "Darcel"))
        self.assertNotEqual(results["Marcel"], results["Darcel"])

    @expectedFailure
    def test_directional_adjective_norp_false_positive(self):
        # §D: "Eastern" (ADJ NORP, in "the Eastern country") is a false positive but
        # is POS-identical to valid demonyms like "Japanese". Cannot be dropped by a
        # POS rule without deleting the demonyms guarded above.
        self.assertFalse(_is_acceptable_entity(FakeSpan("Eastern", ["ADJ"]), NORP))
