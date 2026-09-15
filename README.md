# Gloxy Lexicon Pipeline

Extracting subject-specific vocabulary and rating its difficulty,
for the word pool behind Gloxy (educational quiz game).

## The problem
Gloxy's thesis build used an AI-approximated word list. It worked
well enough to test the systems, but it was the weakest link in
the design: every mechanic assumes the words are well chosen and
correctly rated, and approximation guarantees neither.

## Approach
TF-IDF across per-subject corpora isolates genuinely
subject-specific terms: words frequent in one field's texts but
rare across others. Conjunctions and common verbs score low
everywhere and drop out.

Difficulty combines two axes:
- **Curriculum position** — where a term first appears across
  university-year corpora
- **General frequency** — Zipf frequency via `wordfreq`

The second corrects the first. "Geni" appears late in a biology
curriculum but everyone knows the word; without the frequency
axis it would be rated maximum difficulty.

## What I'd improve
Rating difficulty within a field really needs someone who teaches
it. Curriculum position is a reasonable proxy, not a substitute.

## Note
Source corpora are not included in this repo.
