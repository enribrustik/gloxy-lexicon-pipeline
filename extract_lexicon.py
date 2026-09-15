#!/usr/bin/env python3
"""
Gloxy Lexicon Extractor — prototipo v0.3
=========================================
Estrae candidati lessicali specifici per disciplina da un corpus di testi,
usando lemmatizzazione (spaCy) + TF-IDF (scikit-learn), e ne stima la
difficoltà su due assi.

[#1] FORMA DI SUPERFICIE. Il lemma serve per CONTARE (unisce "obbligazione"
     e "obbligazioni"), ma è brutto da MOSTRARE: spaCy lemmatizza gli
     aggettivi al maschile singolare, producendo "tavola periodico" o
     "politica monetario". Teniamo, per ogni lemma, la forma testuale
     realmente più frequente nel corpus e mostriamo quella.

[#2] REVIEW FLAG. La qualità dell'output è la qualità del corpus: nomi propri
     ed esempi storici (es. "cruzeiro" nella voce sull'inflazione) inquinano
     i candidati. Non possiamo eliminarli automaticamente senza rischiare di
     buttare via termini buoni, ma possiamo SEGNALARLI. La pipeline trova le
     parole; il giudizio resta umano, ma ora è guidato.

[#3] DIFFICOLTÀ A DUE ASSI.
     Asse 1 (corso): se il corpus è organizzato per anno, calcoliamo il
     BARICENTRO degli anni pesato sulle occorrenze. Non l'anno di prima
     apparizione: quello è rumoroso, perché una menzione incidentale in un
     testo di primo anno renderebbe "facile" un termine difficile.
     Asse 2 (notorietà): quanto la parola è comune nell'italiano generale,
     via wordfreq (scala Zipf). Corregge l'asse-corso: "geni" ha baricentro
     alto ma è parola notissima, quindi non è difficile da riconoscere.
     I due assi si combinano in una difficoltà 1-5; ciascuno degrada da solo
     se l'altro non è disponibile.

Struttura del corpus — DUE MODALITÀ
-----------------------------------
A) Piatta (come v0.1, difficoltà non calcolata):
       corpus/Economia/doc1.txt
       corpus/Economia/doc2.txt
B) Per anno (abilita la stima di difficoltà):
       corpus/Economia/anno1/doc1.txt
       corpus/Economia/anno3/doc2.txt
       corpus/Economia/magistrale/doc3.txt
   I nomi-anno riconosciuti sono mappati a un livello numerico (vedi YEAR_MAP).
   Si possono MISCHIARE: una materia per anno, un'altra piatta.

Uso:
    python extract_lexicon.py --top 50
    python extract_lexicon.py --corpus corpus --top 80 --min-df 1

Dipendenze: spacy (+ it_core_news_sm), scikit-learn
"""

import argparse
import csv
import json
import re
import sys
import unicodedata
from collections import defaultdict, Counter
from pathlib import Path

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# [#4] Secondo asse di difficoltà: notorietà nell'italiano generale.
# wordfreq incorpora liste di frequenza offline (incl. OpenSubtitles), quindi
# nessun download di corpora. Se manca, la pipeline continua a funzionare e
# l'asse "notorietà" resta semplicemente disattivato.
try:
    from wordfreq import zipf_frequency
    HAS_WORDFREQ = True
except ImportError:
    HAS_WORDFREQ = False

# ---------------------------------------------------------------------------
# Configurazione
# ---------------------------------------------------------------------------
KEEP_POS = {"NOUN", "PROPN", "ADJ"}
MIN_LEMMA_LEN = 4
TOKEN_RE = re.compile(r"^[a-zàèéìòù]+$")

# [#3] Mappatura nome-cartella -> livello d'anno (intero crescente).
# Aggiungi qui le etichette che usi nel tuo corpus. Tutto ciò che non matcha
# viene trattato come "anno ignoto" e non contribuisce alla difficoltà.
YEAR_MAP = {
    "anno1": 1, "primo": 1, "1": 1, "i": 1,
    "anno2": 2, "secondo": 2, "2": 2, "ii": 2,
    "anno3": 3, "terzo": 3, "3": 3, "iii": 3,
    "magistrale": 4, "anno4": 4, "quarto": 4, "4": 4,
    "specialistica": 5, "anno5": 5, "quinto": 5, "5": 5,
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text


# ---------------------------------------------------------------------------
# 1. PREPROCESSING — lemmatizzazione + raccolta forme di superficie + anni
# ---------------------------------------------------------------------------
def process_corpus(docs, nlp):
    """
    docs: lista di record {materia, year_level, text}.
    Ritorna:
      lemmatized      : {materia: [stringa_di_lemmi_per_documento, ...]}
      surface_counter : {token_lemma: Counter(forma_superficie -> conteggio)}   [#1]
      pos_counter     : {token_lemma: Counter(POS -> conteggio)}                [#2]
      year_occ        : {(materia, token_lemma): Counter(anno -> n_occorrenze)}   [#3]

    'token_lemma' è il token interno (lemma singolo o bigramma con "_").
    """
    lemmatized = defaultdict(list)
    surface_counter = defaultdict(Counter)
    pos_counter = defaultdict(Counter)
    # [#3] Per ogni (materia, token) accumuliamo le occorrenze ripartite per
    # livello d'anno: {(materia, token): Counter(anno -> n_occorrenze)}.
    # Da qui calcoleremo un "baricentro" robusto invece dell'anno minimo.
    year_occ = defaultdict(Counter)

    # Pre-normalizziamo e teniamo i metadati allineati con nlp.pipe
    texts = [normalize(r["text"]) for r in docs]
    metas = [(r["materia"], r["year_level"]) for r in docs]

    for (materia, year_level), doc in zip(metas, nlp.pipe(texts, batch_size=8)):
        tokens = []
        window = []        # buffer (lemma, forma, pos) per i bigrammi contigui
        for tok in doc:
            lemma = tok.lemma_.lower().strip()
            surface = tok.text.lower().strip()
            ok = (
                tok.pos_ in KEEP_POS
                and not tok.is_stop
                and len(lemma) >= MIN_LEMMA_LEN
                and TOKEN_RE.match(lemma)
            )
            if not ok:
                window.clear()
                continue

            tokens.append(lemma)
            surface_counter[lemma][surface] += 1          # [#1]
            pos_counter[lemma][tok.pos_] += 1             # [#2]
            if year_level is not None:                     # [#3]
                year_occ[(materia, lemma)][year_level] += 1

            window.append((lemma, surface, tok.pos_))
            if len(window) == 2:
                bi_lemma = "_".join(w[0] for w in window)
                bi_surface = " ".join(w[1] for w in window)
                tokens.append(bi_lemma)
                surface_counter[bi_lemma][bi_surface] += 1   # [#1] sul bigramma
                # un bigramma è "proprio" se contiene un nome proprio
                bi_pos = "PROPN" if "PROPN" in (w[2] for w in window) else "NOUN"
                pos_counter[bi_lemma][bi_pos] += 1
                if year_level is not None:                   # [#3]
                    year_occ[(materia, bi_lemma)][year_level] += 1
                window.pop(0)

        lemmatized[materia].append(" ".join(tokens))

    return lemmatized, surface_counter, pos_counter, year_occ


def difficulty_from_occ(occ_counter, min_year, max_year):
    """
    [#3] Difficoltà 1–5 dal "baricentro" degli anni, pesato sulle occorrenze.

    Invece dell'anno di PRIMA apparizione (rumoroso: una menzione incidentale
    in un testo di primo anno rende "facile" un termine difficile), usiamo la
    media degli anni pesata su QUANTE volte il termine ricorre in ciascun anno.
    Un termine che esplode alla magistrale ma compare una volta sola in anno1
    avrà baricentro alto -> difficile. Più stabile e fedele all'intuizione.

    Mappiamo poi il baricentro sull'intervallo [min_year, max_year] osservato
    nel corpus, riscalato su 1..5.
    """
    if not occ_counter or max_year is None or max_year == min_year:
        # un solo livello d'anno nel corpus: non c'è gradiente da misurare
        return 1 if occ_counter else None
    total = sum(occ_counter.values())
    centroid = sum(year * n for year, n in occ_counter.items()) / total
    d = 1 + (centroid - min_year) * 4 / (max_year - min_year)
    return max(1, min(5, round(d)))


# ---------------------------------------------------------------------------
# [#4] SECONDO ASSE — notorietà nell'italiano generale (Zipf)
# ---------------------------------------------------------------------------
# Idea: due termini ugualmente "specifici" per il TF-IDF possono avere
# notorietà opposte. "globalizzazione" (Zipf 3.85) la conosce chiunque;
# "signoraggio" (2.80) la conosce solo chi studia economia. La notorietà è
# un asse di difficoltà INDIPENDENTE dall'anno di corso.
#
# La scala Zipf di wordfreq è logaritmica: ~1 = rarissima, ~7 = comunissima
# ("casa" = 5.88). La invertiamo e la mappiamo su 1..5: parola comune ->
# difficoltà bassa; parola assente dal parlato comune -> difficoltà alta.

# Soglie Zipf calibrate sull'italiano: sopra ~4.5 è lessico quotidiano,
# sotto ~2.0 è gergo tecnico/raro. I termini in mezzo scalano linearmente.
ZIPF_EASY = 4.5   # >= questo -> difficoltà 1 (parola comunissima)
ZIPF_HARD = 2.0   # <= questo -> difficoltà 5 (parola rarissima nel parlato)


def fame_difficulty(term_surface):
    """
    Difficoltà 1–5 basata sulla notorietà generale del termine.
    Per i termini multi-parola (bigrammi) usa la parola COMPONENTE più rara:
    un'espressione è "ostica" se contiene anche un solo elemento sconosciuto.
    Ritorna (difficolta, zipf_usata) oppure (None, None) se wordfreq assente.
    """
    if not HAS_WORDFREQ:
        return None, None
    parts = term_surface.split()
    # la componente meno nota domina la percezione di difficoltà
    zipf = min(zipf_frequency(p, "it") for p in parts) if parts else 0.0
    if zipf >= ZIPF_EASY:
        d = 1
    elif zipf <= ZIPF_HARD:
        d = 5
    else:
        # interpolazione lineare inversa tra le due soglie
        d = 1 + (ZIPF_EASY - zipf) * 4 / (ZIPF_EASY - ZIPF_HARD)
        d = max(1, min(5, round(d)))
    return d, round(zipf, 2)


def combine_difficulty(d_year, d_fame):
    """
    Combina i due assi in un'unica difficoltà 1–5.
    - Se abbiamo entrambi: media pesata. Diamo un po' più di peso alla
      notorietà (0.6) perché è il segnale più stabile (il baricentro-anni
      dipende dall'etichettatura del corpus, la notorietà è oggettiva).
    - Se ne abbiamo solo uno, usiamo quello.
    - Se nessuno, None.
    """
    if d_year is not None and d_fame is not None:
        return max(1, min(5, round(0.4 * d_year + 0.6 * d_fame)))
    return d_year if d_year is not None else d_fame


# ---------------------------------------------------------------------------
# [#1] Risoluzione della forma di superficie
# ---------------------------------------------------------------------------
def best_surface(token_lemma, surface_counter):
    """Forma testuale più frequente osservata per quel token. Fallback: lemma."""
    counter = surface_counter.get(token_lemma)
    if not counter:
        return token_lemma.replace("_", " ")
    return counter.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# 2. SCORING — TF-IDF con le materie come "documenti"
# ---------------------------------------------------------------------------
def score_by_materia(lemmatized, surface_counter, pos_counter, year_occ,
                     min_df=1):
    materie = sorted(lemmatized.keys())
    mega_docs = [" ".join(lemmatized[m]) for m in materie]
    n_docs_per_materia = {m: len(lemmatized[m]) for m in materie}

    vectorizer = TfidfVectorizer(
        lowercase=False, token_pattern=r"\S+",
        min_df=min_df, sublinear_tf=True, norm=None,
    )
    matrix = vectorizer.fit_transform(mega_docs)
    vocab = vectorizer.get_feature_names_out()

    # In quante MATERIE appare ogni token (per IDF interpretabile)
    presence = defaultdict(set)
    # In quanti DOCUMENTI di quella materia (per il review flag) [#2]
    doc_freq = defaultdict(lambda: defaultdict(int))
    for i, m in enumerate(materie):
        for doc_str in lemmatized[m]:
            for term in set(doc_str.split()):
                doc_freq[m][term] += 1
        for term in set(mega_docs[i].split()):
            presence[term].add(m)

    # Intervallo di anni osservato nell'intero corpus, per riscalare 1..5 [#3]
    all_years = [y for c in year_occ.values() for y in c]
    min_year = min(all_years) if all_years else None
    max_year = max(all_years) if all_years else None

    results = {}
    for i, materia in enumerate(materie):
        row = matrix.getrow(i).toarray().ravel()
        ranked = sorted(((vocab[j], row[j]) for j in row.nonzero()[0]),
                        key=lambda x: -x[1])
        items = []
        for token_lemma, score in ranked:
            n_materie = len(presence[token_lemma])
            df_here = doc_freq[materia][token_lemma]
            is_propn = pos_counter[token_lemma].most_common(1)[0][0] == "PROPN" \
                if pos_counter.get(token_lemma) else False

            # [#2] review flag: candidati a rischio rumore.
            # Un nome proprio è sospetto soprattutto quando compare in un solo
            # documento (esempio storico spot, es. "cruzeiro"). I PROPN diffusi
            # su più documenti (es. "Parlamento", "Costituzione") sono spesso
            # termini legittimi, quindi non li segnaliamo come rumore.
            single_doc = df_here <= 1 and n_docs_per_materia[materia] > 1
            flags = []
            if single_doc:
                flags.append("nome_proprio_spot" if is_propn else "solo_1_doc")
            review_flag = "|".join(flags) if flags else ""

            # [#3] difficoltà dall'anno di corso (baricentro)
            occ = year_occ.get((materia, token_lemma))
            d_year = difficulty_from_occ(occ, min_year, max_year)
            centroid = (round(sum(y * n for y, n in occ.items()) / sum(occ.values()), 2)
                        if occ else None)

            # [#4] difficoltà dalla notorietà generale (Zipf)
            term_surface = best_surface(token_lemma, surface_counter)
            d_fame, zipf = fame_difficulty(term_surface)

            # difficoltà finale: combinazione dei due assi
            difficulty = combine_difficulty(d_year, d_fame)

            items.append({
                "term": term_surface,                                # [#1]
                "lemma_key": token_lemma.replace("_", " "),
                "score": round(float(score), 3),
                "n_materie": n_materie,
                "is_bigram": "_" in token_lemma,
                "review_flag": review_flag,
                "difficulty": difficulty,            # combinata
                "difficulty_year": d_year,           # asse 1 (corso)
                "difficulty_fame": d_fame,           # asse 2 (notorietà)
                "year_centroid": centroid,
                "zipf": zipf,
            })
        results[materia] = items
    return results


# ---------------------------------------------------------------------------
# 3. I/O
# ---------------------------------------------------------------------------
def load_corpus(corpus_dir):
    """
    Ritorna una lista di record {materia, year_level, text}.
    Riconosce sia la struttura piatta sia quella per-anno (sottocartelle).
    """
    records = []
    for materia_dir in sorted(p for p in corpus_dir.iterdir() if p.is_dir()):
        materia = materia_dir.name
        subdirs = [p for p in materia_dir.iterdir() if p.is_dir()]
        if subdirs:
            # struttura per-anno
            for sub in sorted(subdirs):
                year_level = YEAR_MAP.get(sub.name.lower())
                for f in sorted(sub.glob("*.txt")):
                    records.append({"materia": materia, "year_level": year_level,
                                    "text": f.read_text(encoding="utf-8", errors="replace")})
        # file .txt direttamente nella cartella materia (piatti, anno ignoto)
        for f in sorted(materia_dir.glob("*.txt")):
            records.append({"materia": materia, "year_level": None,
                            "text": f.read_text(encoding="utf-8", errors="replace")})

    materie = {r["materia"] for r in records}
    if len(materie) < 2:
        sys.exit("Servono almeno 2 materie: l'IDF confronta le materie tra loro.")
    return records


def write_csv(out_dir, materia, items):
    csv_path = out_dir / f"candidates_{materia}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "score", "difficulty", "difficulty_year",
                    "difficulty_fame", "zipf", "year_centroid",
                    "n_materie", "is_bigram", "review_flag", "lemma_key"])
        for r in items:
            w.writerow([r["term"], r["score"], r["difficulty"], r["difficulty_year"],
                        r["difficulty_fame"], r["zipf"], r["year_centroid"],
                        r["n_materie"], r["is_bigram"], r["review_flag"], r["lemma_key"]])
    return csv_path


def main():
    ap = argparse.ArgumentParser(description="Gloxy lexicon extractor v0.3")
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--output", default="output")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--min-df", type=int, default=1)
    args = ap.parse_args()

    corpus_dir, out_dir = Path(args.corpus), Path(args.output)
    out_dir.mkdir(exist_ok=True)

    print("Carico il modello italiano di spaCy...")
    nlp = spacy.load("it_core_news_sm", disable=["parser", "ner"])
    nlp.add_pipe("sentencizer")
    nlp.max_length = 2_000_000

    records = load_corpus(corpus_dir)
    materie = sorted({r["materia"] for r in records})
    n_with_year = sum(1 for r in records if r["year_level"] is not None)
    print(f"Corpus: {len(materie)} materie, {len(records)} documenti "
          f"({n_with_year} con livello d'anno -> asse corso"
          f"{' attivo' if n_with_year else ' NON attivo'}).")
    print(f"Asse notorietà (wordfreq): {'attivo' if HAS_WORDFREQ else 'NON attivo (pip install wordfreq)'}.")

    print("Lemmatizzo, raccolgo forme di superficie e livelli d'anno...")
    lemmatized, surface_counter, pos_counter, year_occ = process_corpus(records, nlp)

    print("Calcolo TF-IDF per materia...")
    results = score_by_materia(lemmatized, surface_counter, pos_counter,
                               year_occ, min_df=args.min_df)

    for materia, items in results.items():
        path = write_csv(out_dir, materia, items)
        flagged = sum(1 for r in items if r["review_flag"])
        print(f"  {path}  ({len(items)} termini, {flagged} da revisionare)")

    gloxy = {
        materia: [
            {
                "word": r["term"],
                "macro_topic": materia,
                "tfidf_score": r["score"],
                "difficulty": r["difficulty"],          # [#3+#4] combinata
                "difficulty_year": r["difficulty_year"],  # asse corso
                "difficulty_fame": r["difficulty_fame"],  # asse notorietà
                "zipf": r["zipf"],                        # notorietà grezza (debug)
                "year_centroid": r["year_centroid"],      # baricentro anni (debug)
                "spread_materie": r["n_materie"],
                "review_flag": r["review_flag"],          # [#2] guida la revisione
                "definition_short": "",
                "explanation": "",
                "tags": [],
                "distractors": [],
            }
            for r in items[: args.top]
        ]
        for materia, items in results.items()
    }
    json_path = out_dir / "gloxy_candidates.json"
    json_path.write_text(json.dumps(gloxy, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    print(f"  {json_path}")

    print("\n--- ANTEPRIMA: top 15 per materia ---")
    print("    (D=finale  C=corso  N=notorietà  z=zipf  \u26A0=da revisionare)")
    for materia, items in results.items():
        print(f"\n[{materia}]")
        for r in items[:15]:
            mark = " \u26A0" if r["review_flag"] else ""
            D = r["difficulty"] if r["difficulty"] is not None else "?"
            C = r["difficulty_year"] if r["difficulty_year"] is not None else "-"
            N = r["difficulty_fame"] if r["difficulty_fame"] is not None else "-"
            z = f"{r['zipf']:.1f}" if r["zipf"] is not None else "-"
            print(f"  {r['term']:<26} D{D} (C{C} N{N} z{z}){mark}")


if __name__ == "__main__":
    main()
