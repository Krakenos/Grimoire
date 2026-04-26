# Entity Disambiguation in Grimoire

`filter_similar_entities` (`grimoire/core/grimoire.py`) runs on every request to map surface forms of named entities (as extracted by spaCy) to a single canonical name per entity, so downstream `Knowledge` rows are deduplicated.

## What was fixed (current implementation)

Three issues were observed on larger conversation samples:

**1. Non-transitive grouping**
The original code built a `relation_dict` keyed per entity from raw pairwise matches. If A~B and B~C but A does not directly meet the threshold with C, they'd end up in different clusters depending on which name was the dict key — causing duplicate `Knowledge` rows for the same entity downstream.

Fixed by replacing the per-entity dict with a **Union-Find (Disjoint Set Union)** structure. Every pair meeting the score threshold is unioned, producing true equivalence classes where `result[A] == result[B]` for all members of a cluster.

**2. Over-merging on short names**
The original scorer was `fuzz.partial_ratio`, which returns 100 whenever the shorter string is a substring of the longer one. This caused `"Lin"` → `"Lincoln"`, `"Al"` → `"Alabama"`, etc. to be merged.

Fixed by switching to `fuzz.ratio` (Levenshtein-based, symmetric, length-sensitive). `ratio("lin", "lincoln") ≈ 60`, well below the default threshold of 80.

**3. Cross-label false merges**
Names that fuzzy-match but carry different spaCy entity labels (e.g. `"Banks"` as PERSON vs `"Bank"` as ORG) should not be merged. The label is available for free from the NER pass.

Fixed by partitioning entity names by their most-common spaCy label before clustering. Union is only attempted within the same label group.

**Canonical selection tie-break** (all three fixes combined): shortest name → highest mean intra-cluster score → lexical order. This prefers base/minimal forms over inflected or misspelled variants.

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

### 4. Contextualized token embeddings from the spaCy NER pass (sleeper option)

`en_core_web_trf` is a transformer model. Every token in every document already has a contextualized embedding computed during the NER pass — currently discarded after entity extraction. Each mention of a name at a specific position encodes its local syntactic and semantic context in that vector.

**Why this is interesting:** no additional model inference cost whatsoever. The transformer already ran. Each occurrence of `"Alex"` in a different sentence has a *different* token vector that distinguishes, e.g., "Alex grabbed the sword" from a reference to an entirely different Alex.

**Implementation sketch:**
- In `get_named_entities` (`grimoire/core/grimoire.py`), alongside the `NamedEntity(ent.text, ent.label_)` extraction, also collect `ent.vector` (spaCy exposes the span-level vector from the transformer).
- Thread these per-mention vectors back to the clustering step.
- Average mention vectors per surface form → entity context embedding.
- Use cosine similarity as a secondary filter: only merge a fuzzy-matched pair if their context embeddings also exceed a threshold.

**Caveats:** `ent.vector` availability depends on the pipeline config; may need to ensure the transformer component sets `doc.tensor`. The `en_core_web_trf` pipeline does set token vectors via the transformer component, so this should be available with minor verification.

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

If string-side fixes prove insufficient on real conversation data, the order of effort vs. payoff is:

1. **Contextualized spaCy token embeddings** — zero extra inference cost, meaningful signal for same-message disambiguation.
2. **Coreference resolution** — full solution to the problem, one extra transformer pass.
3. **LLM verification on ambiguous cluster pairs** — highest quality, targeted to only the uncertain cases to control cost.
