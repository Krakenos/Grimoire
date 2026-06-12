# Entity Disambiguation in Grimoire

`filter_similar_entities` (`grimoire/core/entity_filters.py`) runs on every request to map surface forms of named entities (as extracted by spaCy) to a single canonical name per entity, so downstream `Knowledge` rows are deduplicated.

## What was fixed (current implementation)

Seven issues were observed on larger conversation samples (the first three on chat data, the rest on full-epub lorebook generation runs).

**1. Non-transitive grouping**
The original code built a `relation_dict` keyed per entity from raw pairwise matches. If A~B and B~C but A does not directly meet the threshold with C, they'd end up in different clusters depending on which name was the dict key — causing duplicate `Knowledge` rows for the same entity downstream.

Fixed by replacing the per-entity dict with a **Union-Find (Disjoint Set Union)** structure. Every pair meeting the score threshold is unioned, producing true equivalence classes where `result[A] == result[B]` for all members of a cluster.

**2. Over-merging on short names (substring case)**
The original scorer was `fuzz.partial_ratio`, which returns 100 whenever the shorter string is a substring of the longer one. This caused `"Lin"` → `"Lincoln"`, `"Al"` → `"Alabama"`, etc. to be merged.

Fixed by switching to `fuzz.ratio` (Levenshtein-based, symmetric, length-sensitive). `ratio("lin", "lincoln") ≈ 60`, well below the default threshold of 80.

**3. Cross-label false merges**
Names that fuzzy-match but carry different spaCy entity labels (e.g. `"Banks"` as PERSON vs `"Bank"` as ORG) should not be merged. The label is available for free from the NER pass.

Fixed by partitioning entity names by their most-common spaCy label before clustering. Union is only attempted within the same label group.

**4. Over-merging on short strings (single-edit case)**
At the default 80 threshold, `fuzz.ratio` lets in pairs that differ by exactly one character on a short string: `"as"`/`"was"`, `"Le"`/`"let"`, `"an"`/`"and"`, all of which are common spaCy false positives that should never have been entities in the first place but, once present, collapsed into nonsensical clusters.

Fixed by `_passes_threshold` (`grimoire/core/entity_filters.py`): when the shorter string is ≤3 chars **and** at least one side is entirely lowercase, require `match_distance_short` (default 95) instead of `match_distance` (default 80). Capitalised proper nouns of the same length (`Jon`/`John`, `Eve`/`Even`) are unaffected — they still merge at the normal threshold.

**5. Multi-word names that share determiners but differ on the discriminating token**
`"the Nur Empire"` and `"the Loro Empire"` share `the` and `Empire`, so `fuzz.ratio` on the full strings sailed past 80. They are different empires.

Fixed by a discriminating-token gate. After computing the candidate's `_discriminating_tokens` (lowercased word set minus a stopword list `{the, a, an, of, and, de, la, le, los, las}`), if both sides are multi-word and at least one discriminating token is unique to one side, we re-run `fuzz.ratio` on the discriminating-token concatenation alone and require it to also pass `match_distance`. `Nur` vs `Loro` falls well below 80 on its own, so the merge is blocked. `the Royal Academy`/`Royal Academy` is unaffected because the discriminating sets are identical (`{royal, academy}`).

**6. Whitespace and possessive variants creating phantom duplicates**
`Cooley III's`, `the Bauer Kingdom's`, `the  Bauer Kingdom` (double-space from html parsing) were arriving as distinct surface forms. They merge correctly *most* of the time but at the cost of a fuzzy-match step and noisy canonical selection.

Fixed at the source: `_clean_entity_text` (NFKC, NBSP→space, whitespace collapse, trailing `'s` strip, leading/trailing punctuation strip) is applied at `NamedEntity` construction time in both `get_named_entities` and `generate_lorebook`. The text extractors (`extract_text_from_epub`, `extract_text_from_pdf`) also feed their output through `_normalize_text` so spaCy never sees the embedded double-spaces in the first place.

**7. Canonical selection naming the corrupted variant**
The original tie-break was *shortest name → highest mean intra-cluster score → lexical order*. On the epub run this systematically chose the **wrong** representative whenever a cluster contained a misspelled/de-accented/truncated form, because such forms tend to be shorter or sort earlier:

- `Clare`/`Claire` → canonical `Clare` (5 < 6 chars; but `Claire` is the protagonist)
- `Claire Francois`/`Claire François` → canonical `Claire Francois` (`c` < `ç` lexically)
- `Darcel`/`Marcel` (a genuine over-merge, see residuals) → canonical picked by length/lexical rather than the dominant spelling

Fixed by making **surface-form frequency the primary tie-break key**: `(-count, length, -mean_score, lexical)`. The correct spelling almost always dominates a typo across a full document, so the most-seen form wins. `filter_similar_entities` takes an optional `name_counts: dict[str, int]`; both call sites pass a `Counter` over the *non-deduplicated* mention stream (`chain(*entity_list)`) — note this is true mention frequency, distinct from the `unique_ents`-derived label counts. Omitting `name_counts` (or passing equal counts) falls back to the previous length/lexical behaviour, so existing callers and tests are unaffected.

## Companion: spaCy false-positive filtering

A separate (but adjacent) problem visible on the lorebook sample was that `en_core_web_trf` mislabels a substantial number of common nouns and function words as named entities (`danger`, `noble`, `place`, `host`, `and`, `or`, `by`, `the`, `let`, `Q18`, etc.). These never reach the disambiguation step in the current code:

- **`BANNED_LABELS`** (`DATE`, `CARDINAL`, `ORDINAL`, `TIME`, `QUANTITY`, `PERCENT`, `MONEY`) — single source of truth shared by both pipelines (replacing two duplicated inline lists).
- **`TRIVIAL_ENTITY_TEXTS`** — hand-curated set of verbs/adjectives that spaCy mislabels even with content POS (`danger`, `ran`, `shamed`, `without`).
- **POS gate** — any entity span whose every token is `DET/ADP/AUX/PRON/CCONJ/SCONJ/PART/INTJ` is rejected. Catches `and`, `or`, `by`, `she`, `let` and similar without an explicit list.
- **Lowercase-single-token rule** — single-token, fully-lowercase, ≤6 chars is rejected. Catches `noble`, `host`, `place`, `top`, `modern`.
- **`WORK_OF_ART` is the noisiest label** — require multi-token + initial capital. Keeps real titles (`The Hateful Cry`, `Seven Seas`); drops `ran`, `world`, `to`, `Q18`.

These filters apply at NamedEntity construction time, so neither the disambiguation step nor downstream `Knowledge` persistence ever sees the false positives.

---

## Known residual limitations (second-run analysis)

After the §1–§7 fixes, a second full-epub run was inspected (128 lorebook entries; a per-span POS/accept-reject debug dump confirmed the regression picture). The filters are well-calibrated — of ~3,260 candidate spans, ~557 were rejected and **none** were legitimate named entities (the rejections were all banned numeric/temporal labels plus clear function-word FPs). The remaining oddities are all cases that **string- and POS-side methods cannot fix without unacceptable collateral**, recorded here so they aren't re-investigated from scratch:

**A. Single-substitution over-merge on medium-length names — irreducible.**
`Marcel` was merged into `Darcel` (`fuzz.ratio = 83.3`, both 6 chars and capitalised, so the short-string gate in §4 doesn't apply). There is no safe string threshold: legitimate near-pairs in the *same* book score as high or higher — `Clare`/`Claire` = 90.9, `Mil`/`Mils` = 85.7. Separating "same name, one typo" from "two different names one letter apart" requires semantic/discourse signal (coreference or LLM, §5/§7). Left as-is.

**B. Under-merging on word-order / extra-token variants.**
`fuzz.ratio` is order- and length-sensitive, so these stayed split when they're one entity:

| Split entries | `ratio` | `token_set_ratio` |
|---|---|---|
| `The Loro Empire` / `the Empire of Loro` | 60.6 | 100 |
| `Rei Ohashi` / `Oohashi Rei` / `Rei` | 57.1 | 95 |
| `Academy` / `the Academy` / `Royal Academy` | 70–78 | 100 |
| `The Aurousseau Company` / `the Aurousseau Commercial Firm` | 73.1 | 78 |

`token_set_ratio`/`token_sort_ratio` would recover most of these, **but** they also score `Royal Academy` vs `Academy` and `Academy Knights` vs `Academy` at 100 — collapsing distinct institutions. This is a genuine precision/recall trade-off, not a free win; the scorer was deliberately **not** switched. Revisit only with a method that can tell "subset name" (merge) from "distinct sibling institution" (keep) apart — again a semantic question.

**C. spaCy span-boundary false positives.**
- Comma/`and`-joined name lists fused into one entity: `Rod Yu Thane`, `Rod Yu Thane Claire`.
- Truncated/dangling spans: `Great Famine of`, `My Lady, Claire François Afterword` (a chapter heading).
- Stylised duplicates of real entities: `I-Ingrate!`, `Claaaaaire`.

These are NER extent errors; a generic filter risks dropping real multi-word names, so they're left to the long tail.

**D. Adjectival / directional NORP — POS cannot separate good from bad.**
`Eastern` (`ADJ/JJ/amod`, in "the Eastern country") is a false positive, but it is **POS-identical** to legitimate demonyms accepted in the same run — `Japanese` (×12), `English`, `Greek`, `Buddhist`, `Alpecian` are all `ADJ/JJ/amod` `NORP`. A "single-token ADJ NORP → drop" rule would delete 6+ valid entities to kill one FP. Not worth a rule; a small curated directional-adjective stoplist is the only safe lever, and the cost of leaving it (one spurious entry) is lower.

**E. Surname / family-name metonymy — inherently ambiguous.**
Bare `François` (22 mentions) is correctly extracted and correctly kept distinct from `Claire François`, `Dole François`, and `House François`. But the bare surname alone refers sometimes to a person, sometimes to the house — genuinely ambiguous even to a human without context. No automated method short of full coreference resolves it, and the entry is arguably useful as "House of François" regardless. Left as-is.

The common thread: every residual is a *semantic* discrimination problem wearing a string-similarity costume. The string/POS tier has reached its ceiling; further gains require §5 (coreference) or §7 (LLM verification).

---

## Deferred: semantic / contextual approaches

The following approaches were considered and deferred. They address a different axis of the problem — disambiguation based on *meaning and context* rather than *spelling*. They become relevant if the string-side fixes are insufficient.

### 1. Re-embedding entity names (not recommended)

Embed entity name strings with the existing `gte-base-en-v1.5` model and use cosine similarity as an additional gate.

**Why not:** Sentence-transformers are trained for semantic similarity of phrases and sentences, not orthographic closeness of short proper nouns. `"Alex"` and `"Alexei"` may land far apart while unrelated names with similar phonology land close. Worse signal-to-noise than pure Levenshtein for this use case. Adds Redis lookups per new name.

### 2. Context signature from existing message/knowledge embeddings

For each candidate entity in the current batch, build a context vector by averaging the `Message.vector_embedding` values of messages it appears in. Compare against `Knowledge.vector_embedding` (the summary embedding, set by the `describe_entity` Celery task) for entities already in the DB.

**Cost:** near-zero — message embeddings are already computed per request and knowledge embeddings are already in the DB.

**Why deferred:** The dominant failure pattern is *new* entities that co-occur in the same small batch of messages. Because they appear together, their context vectors are near-identical — the signal vanishes precisely where disambiguation is hardest. The approach is more useful for comparing a new candidate against a well-established existing entity, which is a narrower case.

**Precondition if implemented:** need to pass `chat_id` and the per-message entity map through to the clustering step.

### 3. Sentence-level / span-level embeddings

Instead of message-level context, embed individual sentences or a ±N-token window around each entity mention.

**Improvement over #2:** two entities in the same message but different sentences get distinct context vectors. Same-sentence co-occurrences still collapse.

**Cost:** additional encode calls proportional to sentence/span count per message. Cheap if batched; requires plumbing sentence boundaries or token offsets.

### 4. Contextualized token embeddings from the spaCy NER pass — *empirically ineffective*

This was previously listed as the "sleeper option": no extra inference cost, the transformer already ran, just compare span vectors. **It was prototyped end-to-end and removed**. The reasoning and the data:

**Vector availability.** `Span.vector` returns an empty array on `en_core_web_trf` because the pipeline doesn't populate `doc.tensor` (curated-transformers stores embeddings on a `Doc` extension, not on the standard tensor field). The actual transformer hidden states are reachable via:

```python
trf = doc._.trf_data
data    = trf.last_hidden_layer_state.data     # (n_wordpieces, 768)
lengths = trf.last_hidden_layer_state.lengths  # wordpieces per spaCy token
offsets = np.concatenate([[0], np.cumsum(lengths)])
# Span vector for tokens [i, j):
vec = data[offsets[i]:offsets[j]].mean(axis=0)
```

So the data is there. The startup check that was originally proposed for `Span.vector` returns `False`; using `_.trf_data.last_hidden_layer_state` works.

**Why the gate was removed anyway.** The cosine similarity of last-hidden-layer means is dominated by **syntactic position**, not lexical identity. Measured on a small narrative sample:

| Pair | Same entity? | Raw cos | Mean-centered cos |
|---|---|---|---|
| Marcel(1) vs Marcel(2) | yes | 0.53 | **0.27** |
| Marcel vs Darcel | no | **0.99** | 0.98 |
| Claire(1) vs Claire(2) (parallel sentences) | yes | 1.00 | 1.00 |
| Yu Bauer vs Rod Bauer | no | **0.99** | 0.97 |
| Yu Bauer vs Bauer | partial | 0.68 | 0.45 |
| The Nur Empire vs the Loro Empire | no | 0.86 | 0.77 |

The signal goes the **wrong direction**: a pair of true Marcel mentions in different syntactic roles (0.27 mean-centered) is *less similar* than the false-merge pair Marcel/Darcel (0.98). No threshold over this distribution can separate same-entity from different-entity pairs without massive collateral. Mean-centering, max-pooling instead of mean-pooling, dropping shared tokens — none recovered useful separation. Earlier hidden layers aren't exposed by the curated-transformers pipeline (`all_hidden_layer_states` is empty).

**Why this happens.** This is the well-documented *anisotropy* of transformer hidden states: representations are concentrated in a narrow cone of vector space, and within that cone "subject of a past-tense verb" dominates "the proper noun is X". The transformer was trained for tagging/parsing/NER decisions, not for an identity discrimination task between *which* proper noun is in the slot. For that you'd need a model trained on the disambiguation objective directly — see options #5 (coreference) and #7 (LLM verification).

**Take-away.** Don't reinstate this without changing the scoring approach. Possible angles if revisited: a sentence-piece-level lexical anchor (the wordpiece IDs themselves are a stronger identity signal than their hidden states), or piping spans through a small dedicated similarity head. Until then, the string-side gates carry the disambiguation work and the vector path is dead code.

### 5. Coreference resolution

Dedicated models that answer "do these two mentions refer to the same entity in this discourse" — the actual ML task behind disambiguation.

Candidate libraries: `fastcoref`, `coreferee`, spaCy's `experimental-coref` component.

**Quality:** substantially better than any string-similarity approach for hard cases (pronoun chains, relational references like "Mr. Smith"/"John"/"Dad").

**Cost:** roughly another transformer pass per request (comparable to the NER pass). Significant but not prohibitive if `prefer_gpu` is set.

**Fit for Grimoire:** coreference works best on continuous narrative text, which matches chat logs well. However it's designed for within-document resolution; cross-message resolution over a long chat history is still challenging.

### 6. Entity linking against a knowledge base (Wikidata/Wikipedia)

Map each entity mention to a KB entry. Two mentions resolving to the same KB entry are the same entity.

**Best for:** public figures, locations, organizations with Wikipedia entries.

**Worst for:** original characters in fiction — which is likely the majority of Grimoire's traffic. Not recommended as a primary approach.

### 7. LLM-based clustering pass

Pass candidate entity names + surrounding context windows to an LLM and ask it to cluster co-referential mentions.

**Quality ceiling:** highest of all approaches. Handles relational coreference ("he"/"the king"/"Arthur"), abbreviations, nicknames.

**Cost:** an additional LLM call per request, on top of the summarization tasks already queued. Latency and cost are significant.

**Reasonable use:** as a post-processing step on *ambiguous* clusters (pairs that score between, say, 60–80 on the string scorer) rather than all entity pairs, to keep the call small and targeted.

---

## Recommended upgrade path

The string-side fixes (§1–§7) are the pragmatic tier, and the second-run analysis above shows they've reached their ceiling — the remaining errors are all semantic. If they prove insufficient on richer conversation data, the order of effort vs. payoff is:

1. **Coreference resolution** — full solution to the problem, one extra transformer pass.
2. **LLM verification on ambiguous cluster pairs** — highest quality, targeted to only the uncertain cases (e.g. fuzzy score in 60–80) to keep cost bounded.

The previously-listed "contextualized spaCy embeddings" option is empirically a dead end; see §4 above.

---

## Where disambiguation runs: request-time vs. eventual consistency

A natural objection to the semantic tier (§5/§7) is: *for chats, disambiguation happens at request time — the client is blocked waiting for the response — so an LLM call can't be queued like summarization is.* This is half right, and the distinction matters for any future implementation.

**What is actually synchronous in `process_request`.** The client waits for `semantic_search(...)`, which ranks **previously-stored** `Knowledge` rows by cosine similarity to the current messages. The synchronous work is: NER on new messages → `filter_similar_entities` (the fuzzy dedup, sub-100ms) → create/link `Knowledge` rows → retrieval. The LLM that writes each row's **summary text and embedding** (`describe_entity`) is *already* deferred to Celery.

**Why canonicalization is a write-path decision, not a read-path one.** A `Knowledge` row created during a request has **no summary and no embedding until `describe_entity` runs**. Because retrieval ranks by embedding, a brand-new row **cannot be returned by the response that created it**. Therefore a wrong cluster decision in the current request has *zero* effect on the current response — its only cost is polluting *future* retrievals, and there is time to fix it before the next relevant request.

**Consequence: verification can be eventually-consistent.**

- *Synchronous (unchanged):* the cheap fuzzy dedup writes a *provisional* canonical assignment. Good enough, because the rows it touches aren't retrievable yet anyway.
- *Async (same Celery lane as `describe_entity`):* the expensive verifier (LLM on ambiguous pairs, or coref) **reconciles** — merges a wrongly-split pair, or splits a wrongly-merged one (e.g. `Marcel`/`Darcel`) — after the fact. The correction benefits every subsequent request.

The client never blocks. This mirrors exactly how summary *text* is already handled: a fast provisional pass synchronously, the correct pass in the background.

**Where the cost moves: latency → reconciliation complexity.** The hard part is no longer compute, it's that async merge/split of existing `Knowledge` rows is harder than creating them: merging means reattaching `Message`↔`Knowledge` links, combining (or regenerating) the two summaries, re-embedding, and deleting the loser — while possibly racing an in-flight `describe_entity` for the row being merged; splitting additionally requires re-partitioning the linked messages. The data model already supports this (`update_count`, the message-knowledge links, `frozen`/`enabled` flags), so it is tractable — but it is the honest price of the semantic tier on the chat path.

**The one case that forces a synchronous verifier.** If eventual consistency is rejected and disambiguation must be *final* at request time, then **coreference beats an LLM on the chat path specifically** — it is a bounded, local transformer pass (tens of ms on GPU, no network round-trip), whereas an LLM generation call is unbounded and network-bound to the summarization backend. This reverses the general §7-over-§5 preference, but only under the synchronous-correctness constraint, and coref still only partially fixes the residuals (helps metonymy, not the single-substitution string collisions).

**Cost anchor (measured, GPU).** `en_core_web_trf` runs ≈ 13,700 tok/s / ≈ 620 short-docs/s on the dev GPU. A coref pass is the same order of magnitude (≈ 1–3× the NER pass). On chat batches the absolute synchronous cost is a few ms; the caveats are that the default config ships `prefer_gpu: False` (CPU `trf` is ~10–30× slower, so coref roughly doubles an already-slow NLP step) and that coref is within-document only — it does not link to `Knowledge` rows from prior requests, which is part of what the chat path actually needs.
