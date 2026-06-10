"""Entity false-positive filtering and name disambiguation.

This is the string/POS tier of Grimoire's entity handling: it decides which
spaCy spans are plausible named entities (`_is_acceptable_entity`) and maps
surface-form variants of the same entity to a single canonical name
(`filter_similar_entities`). All functions here are pure (no DB, no NLP model,
no Redis) and operate on strings, spaCy spans, or score matrices.

See docs/entity_disambiguation.md for the rationale behind every rule and
constant below — they are empirically calibrated, not arbitrary.
"""

import re
import unicodedata
from collections import defaultdict
from itertools import combinations

import numpy as np
from rapidfuzz import fuzz
from rapidfuzz import process as fuzz_process
from rapidfuzz import utils as fuzz_utils

from grimoire.core.settings import settings

BANNED_LABELS = frozenset({"DATE", "CARDINAL", "ORDINAL", "TIME", "QUANTITY", "PERCENT", "MONEY"})

# Common words that en_core_web_trf occasionally mislabels as named entities.
# The POS gate catches most of these; this set catches verbs/adjectives that
# receive content POS tags despite being clear non-entities.
TRIVIAL_ENTITY_TEXTS = frozenset(
    {
        "a",
        "an",
        "and",
        "as",
        "at",
        "be",
        "been",
        "being",
        "both",
        "by",
        "danger",
        "for",
        "got",
        "grinned",
        "had",
        "has",
        "have",
        "her",
        "him",
        "his",
        "if",
        "is",
        "it",
        "its",
        "laughed",
        "let",
        "made",
        "make",
        "of",
        "on",
        "or",
        "out",
        "ran",
        "run",
        "shamed",
        "shook",
        "shrill",
        "so",
        "sure",
        "that",
        "the",
        "them",
        "they",
        "this",
        "to",
        "us",
        "was",
        "we",
        "without",
        "yawn",
        "yawned",
        "you",
    }
)

# spaCy POS tags for closed-class / function words — any entity span composed
# entirely of these is a false positive.
NON_CONTENT_POS = frozenset({"DET", "ADP", "AUX", "PRON", "CCONJ", "SCONJ", "PART", "INTJ"})

# Tokens to ignore when computing the "discriminating" token set for multi-word
# entity names (used in filter_similar_entities to block cross-name merges).
DROP_TOKENS = frozenset({"the", "a", "an", "of", "and", "de", "la", "le", "los", "las"})


def _clean_entity_text(s: str) -> str:
    """Normalize a spaCy entity surface form: collapse whitespace, strip possessives and punctuation."""
    s = unicodedata.normalize("NFKC", s).replace(" ", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"['''’]s$", "", s)  # strip trailing possessive 's / 's
    s = s.strip("'\"`‘’“”.,;:()[]{}—–-")
    return s


def _is_acceptable_entity(span, label: str) -> bool:
    """Return True if the spaCy entity span is a plausible named entity worth keeping."""
    cleaned = _clean_entity_text(span.text)
    if len(cleaned) < 2:
        return False
    if label in BANNED_LABELS:
        return False
    if cleaned.lower() in TRIVIAL_ENTITY_TEXTS:
        return False
    # Spans made entirely of function/closed-class words are not entities.
    if all(tok.pos_ in NON_CONTENT_POS for tok in span):
        return False
    # Single-token, fully-lowercase, short strings are almost always common-noun FPs.
    if " " not in cleaned and cleaned == cleaned.lower() and len(cleaned) <= 6:
        return False
    # WORK_OF_ART is very noisy; require at least two tokens and an initial capital.
    if label == "WORK_OF_ART" and (" " not in cleaned or not cleaned[0].isupper()):
        return False
    return True


def _discriminating_tokens(s: str) -> set[str]:
    """Return the non-stopword tokens of a (possibly multi-word) entity name."""
    return {t for t in re.findall(r"\w+", s.lower()) if t not in DROP_TOKENS}


def _passes_threshold(name_a: str, name_b: str, score: float) -> bool:
    """Apply a stricter threshold for very short strings.

    Proper nouns are normally capitalized; function-word FPs are often all-lowercase
    (even mid-sentence) or differ by exactly one letter from a common lowercase word.
    Apply the strict threshold only when the shorter string is ≤3 chars AND at least
    one side is entirely lowercase, which covers the main collision patterns (as/was,
    Le/let, an/and) without penalizing short proper names like Jon/John or Alex/Alexy.
    """
    min_len = min(len(name_a), len(name_b))
    if min_len <= 3 and (name_a == name_a.lower() or name_b == name_b.lower()):
        return score >= settings.match_distance_short
    return score >= settings.match_distance


def _should_merge(name_a: str, name_b: str, label_a: str | None, label_b: str | None, score: float) -> bool:
    """Decide whether two entity surface forms are variants of the same entity.

    Three gates, applied in order (see docs/entity_disambiguation.md §3–§5):
      1. same spaCy label;
      2. length-aware Levenshtein threshold (short names need near-identity);
      3. for multi-word names whose discriminating (non-stopword) tokens diverge,
         the scorer applied to those tokens alone must also pass — this blocks
         e.g. "the Nur Empire" / "the Loro Empire", which share "the"/"Empire".
    """
    if label_a != label_b:
        return False
    if not _passes_threshold(name_a, name_b, score):
        return False
    da = _discriminating_tokens(name_a)
    db = _discriminating_tokens(name_b)
    if len(da) > 1 and len(db) > 1:
        overlap = da & db
        if len(overlap) < min(len(da), len(db)):
            token_score = fuzz.ratio(" ".join(sorted(da)), " ".join(sorted(db)))
            if token_score < settings.match_distance:
                return False
    return True


def filter_similar_entities(
    entity_names: list[str],
    entity_labels: dict[str, str],
    name_counts: dict[str, int] | None = None,
) -> dict[str, str]:
    if not entity_names:
        return {}

    counts = name_counts or {}

    n = len(entity_names)
    result_matrix = fuzz_process.cdist(
        entity_names,
        entity_names,
        scorer=fuzz.ratio,
        processor=fuzz_utils.default_process,
    )

    # Union-Find with path compression
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i, j in combinations(range(n), 2):
        name_i, name_j = entity_names[i], entity_names[j]
        if _should_merge(name_i, name_j, entity_labels.get(name_i), entity_labels.get(name_j), result_matrix[i][j]):
            union(i, j)

    # Group member indices by cluster root
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        clusters[find(i)].append(i)

    results = {}
    for members in clusters.values():
        if len(members) == 1:
            name = entity_names[members[0]]
            results[name] = name
            continue
        # Rank candidates: most frequent surface form first (the correct spelling
        # almost always dominates a misspelled/de-accented/truncated variant), then
        # shortest, then highest mean intra-cluster score, then lexical.
        ranked = sorted(
            ((entity_names[i], np.mean([result_matrix[i][j] for j in members])) for i in members),
            key=lambda c: (-counts.get(c[0], 0), len(c[0]), -c[1], c[0]),
        )
        canonical = ranked[0][0]
        for i in members:
            results[entity_names[i]] = canonical

    return results
