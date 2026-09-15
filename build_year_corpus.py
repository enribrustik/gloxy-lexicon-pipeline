#!/usr/bin/env python3
"""
build_year_corpus.py — corpus di prova ORGANIZZATO PER ANNO DI CORSO.
=====================================================================
Dimostra il fix #3: la difficoltà di un termine stimata dall'anno più
precoce in cui compare. Scarica voci Wikipedia (CC BY-SA) e le dispone in
sottocartelle anno1/anno2/magistrale per due materie.

La logica della scelta: voci "introduttive" vanno in anno1, voci "avanzate"
in anni superiori. Così un termine come "prezzo" (presente già in anno1)
risulta facile, mentre "derivato"/"signoraggio" (solo magistrale) risulta
difficile.

Uso: python build_year_corpus.py
"""
import json, time, urllib.parse, urllib.request
from pathlib import Path

# materia -> { etichetta_anno -> [voci Wikipedia] }
PLAN = {
    "Economia": {
        "anno1": ["Mercato (economia)", "Domanda e offerta", "Prezzo", "Moneta"],
        "anno2": ["Inflazione", "Prodotto interno lordo", "Politica monetaria"],
        "magistrale": ["Strumento finanziario derivato", "Signoraggio",
                       "Politica fiscale"],
    },
    "Biologia": {
        "anno1": ["Cellula", "Membrana cellulare", "Organulo"],
        "anno2": ["Fotosintesi clorofilliana", "Mitosi", "Enzima"],
        "magistrale": ["Espressione genica", "Apoptosi", "Epigenetica"],
    },
}

API = "https://it.wikipedia.org/w/api.php"


def fetch(title):
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "format": "json", "formatversion": 2, "redirects": 1, "titles": title})
    req = urllib.request.Request(f"{API}?{params}",
        headers={"User-Agent": "GloxyLexiconPrototype/0.2 (tesi)"})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", [])
            return pages[0].get("extract") if pages else None
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(6 * (attempt + 1)); continue
            return None
        except Exception:
            return None
    return None


def main():
    root = Path("corpus_anni")
    for materia, years in PLAN.items():
        for year, titles in years.items():
            out = root / materia / year
            out.mkdir(parents=True, exist_ok=True)
            for title in titles:
                safe = title.replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")
                dest = out / f"{safe}.txt"
                if dest.exists():
                    print(f"  [skip] {materia}/{year}/{dest.name}"); continue
                text = fetch(title)
                if text and len(text) > 800:
                    dest.write_text(text, encoding="utf-8")
                    print(f"  [ok]   {materia}/{year}/{dest.name} ({len(text)} c.)")
                else:
                    print(f"  [!!]   {materia}/{year}: '{title}' non trovata")
                time.sleep(2)
    print("\nCorpus per-anno pronto in ./corpus_anni")
    print("Ora: python extract_lexicon.py --corpus corpus_anni --top 40")


if __name__ == "__main__":
    main()
