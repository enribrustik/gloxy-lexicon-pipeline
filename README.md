# Gloxy Lexicon Extractor

Extracts subject-specific vocabulary from a corpus and estimates how difficult
each term is, on two axes. Built to replace the AI-approximated word pool behind
[Gloxy](https://enricomrustici.notion.site/Enrico-M-Rustici-Game-Design-Portfolio-3d77d2714c7a8096bed5ee2ec9d38e5b),
an educational quiz game where the atomic unit of gameplay is the word.

*Code comments and console output are in Italian; the pipeline targets Italian
corpora and uses an Italian spaCy model.*

## The problem

Gloxy scores a point when a player recognises a word from its definition. That
makes two things load-bearing: the words have to be genuinely characteristic of
their subject, and their difficulty ratings have to be right. The thesis build
approximated both with an LLM. It worked well enough to test the game systems,
but it was the weakest link in the design, because every mechanic rests on
assumptions the approximation couldn't guarantee.

This pipeline replaces guessing with measurement.

## Finding subject-specific words

Terms are lemmatised with spaCy, then scored with TF-IDF across per-subject
corpora. A word frequent in one field's texts but rare across the others is
characteristic of that field; conjunctions and common verbs score low everywhere
and drop out on their own.

Two refinements matter in practice:

**Readable surface forms.** Lemmas are right for *counting* (they merge
"obbligazione" and "obbligazioni") but wrong for *display*: spaCy lemmatises
adjectives to masculine singular, producing "politica monetario". The pipeline
keeps the most frequent surface form actually observed and shows that instead,
so the output reads "politica monetaria", "tavola periodica".

**Review flags rather than silent deletion.** Proper nouns and one-off historical
examples pollute the candidate list. They can't be filtered automatically without
discarding good terms, so the pipeline flags them (`solo_1_doc`,
`nome_proprio_spot`) and leaves the judgement to a human who now knows where to
look. Widely-distributed proper nouns like "Costituzione" are *not* flagged,
because they're real terms.

## Rating difficulty on two axes

Difficulty isn't one thing. Two words equally specific to a subject can be very
different to recognise:

|  | **Common in general Italian** | **Rare in general Italian** |
|---|---|---|
| **Studied early** | easy (*cellula*) | medium (*enzima*) |
| **Studied late** | medium (*reddito*) | hard (*signoraggio*) |

**Axis 1, curriculum position.** If the corpus is organised by year of study,
the pipeline computes the weighted centroid of the years in which a term occurs.
Deliberately *not* the year of first appearance: that's noisy, because a passing
mention in a first-year text would rate a genuinely advanced term as easy.

**Axis 2, general notoriety.** How common the word is in everyday Italian, via
`wordfreq`'s Zipf scale. Objective and requires no labelling.

The second axis corrects the first, which is the point. "Geni" has a high
curriculum centroid because it appears in advanced texts, but everyone knows the
word, so its final rating drops from 5 to 3. "Reddito" goes from 4 to 2 the same
way. Without the notoriety axis these would be rated as hard as "signoraggio",
which they plainly aren't.

Final difficulty is a weighted combination (0.4 curriculum, 0.6 notoriety;
notoriety weighs more because it's the more stable signal). Both sub-scores stay
visible in the output for transparency.

**Graceful degradation:** without a year-labelled corpus, notoriety runs alone;
without `wordfreq`, the curriculum axis runs alone; without either, `difficulty`
is `null`. No crashes in any case.

## Setup

```bash
pip install spacy scikit-learn wordfreq
python -m spacy download it_core_news_sm   # it_core_news_lg gives better results
```

`wordfreq` is optional but recommended: it enables the second difficulty axis.

## Usage

```bash
# Flat corpus (no curriculum axis)
python build_demo_corpus.py
python extract_lexicon.py --top 50

# Year-labelled corpus (enables both axes)
python build_year_corpus.py
python extract_lexicon.py --corpus corpus_anni --output output_anni --top 40
```

Both corpus builders pull plain text from Italian Wikipedia (CC BY-SA) via the
official API. They exist so the pipeline is runnable end to end; a real corpus
would use open-access course materials, public syllabi and open theses.

### Corpus structure

```
# Flat:
corpus/Economia/inflazione.txt

# Year-labelled:
corpus_anni/Economia/anno1/prezzo.txt
corpus_anni/Economia/anno2/inflazione.txt
corpus_anni/Economia/magistrale/signoraggio.txt
```

Recognised year labels: `anno1/primo/i`, `anno2/secondo/ii`, …, `magistrale`,
`specialistica` (see `YEAR_MAP`). Structures can be mixed per subject.

## Output

- `candidates_<Subject>.csv` — full ranking with `term, score, difficulty,
  year_centroid, n_materie, is_bigram, review_flag, lemma_key`. Sortable and
  filterable in a spreadsheet.
- `gloxy_candidates.json` — top-N per subject with difficulty pre-filled, ready
  for the authoring stage.

## Known limits

**Over-flagging on small corpora.** With few documents per subject, nearly every
term trips `solo_1_doc`. The flag only becomes selective with dozens of documents
per subject. That's a limitation of the data, not the algorithm, and I haven't
masked it with an arbitrary threshold.

**Difficulty depends on corpus labelling.** The curriculum axis is only as good
as the assignment of texts to years. That's a human input, but made once per
document rather than once per word, so the gain in scale is still large.

**Thresholds are uncalibrated.** `ZIPF_EASY`, `ZIPF_HARD` and the combination
weights were set by judgement. Validating them needs a few hundred words already
rated by hand as ground truth.

**The real gap is expertise.** Rating how hard a term is within its own field
properly requires someone who teaches it. Curriculum position is a reasonable
proxy, not a substitute.

## A note on corpora

The pipeline only ever saves word lists and statistics, never passages of source
text. No corpora are included in this repository.
