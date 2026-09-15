#!/usr/bin/env python3
"""
build_demo_corpus.py — costruisce un corpus di prova da Wikipedia italiana.
============================================================================
Scarica il testo (plain text, licenza CC BY-SA — legalmente sereno per
l'analisi) di alcune voci per materia, tramite l'API ufficiale di Wikipedia.

È solo per il bootstrap: per il prodotto vero il corpus va arricchito con
dispense open access, sillabi CISIA, tesi pubbliche, ecc. La struttura
delle cartelle resta identica, quindi la pipeline non cambia.

Uso:
    python build_demo_corpus.py
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Voci per materia: scegline di "centrali" per la disciplina. Più voci =
# statistiche migliori. Modifica liberamente questo dizionario.
SEED = {
    "Economia": [
        "Inflazione", "Mercato finanziario", "Microeconomia",
        "Bilancio d'esercizio", "Politica monetaria",
    ],
    "Biologia": [
        "Cellula", "Fotosintesi clorofilliana", "DNA",
        "Evoluzione", "Sistema immunitario",
    ],
    "Giurisprudenza": [
        "Diritto privato", "Contratto", "Costituzione della Repubblica Italiana",
        "Processo penale", "Proprietà (diritto)",
    ],
    "Chimica": [
        "Entalpia", "Legame chimico", "Termodinamica chimica",
        "Reazione chimica", "Tavola periodica degli elementi",
    ],
}

API = "https://it.wikipedia.org/w/api.php"


def fetch_extract(title: str) -> str | None:
    params = urllib.parse.urlencode({
        "action": "query", "prop": "extracts", "explaintext": 1,
        "format": "json", "formatversion": 2, "redirects": 1,
        "titles": title,
    })
    req = urllib.request.Request(
        f"{API}?{params}",
        headers={"User-Agent": "GloxyLexiconPrototype/0.1 (progetto di tesi)"},
    )
    # Retry con backoff: l'API di Wikipedia risponde 429 se si insiste troppo.
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            pages = data.get("query", {}).get("pages", [])
            if pages and "extract" in pages[0]:
                return pages[0]["extract"]
            return None
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                wait = 5 * (attempt + 1)
                print(f"         (429: aspetto {wait}s e riprovo...)")
                time.sleep(wait)
            else:
                print(f"         (errore HTTP {e.code}, salto la voce)")
                return None
        except Exception as e:
            print(f"         (errore: {e}, salto la voce)")
            return None
    return None


def main():
    root = Path("corpus")
    for materia, titles in SEED.items():
        out = root / materia
        out.mkdir(parents=True, exist_ok=True)
        for title in titles:
            safe = title.replace(" ", "_").replace("/", "-").replace("'", "")
            dest = out / f"{safe}.txt"
            if dest.exists():
                print(f"  [skip] {materia}/{dest.name}")
                continue
            text = fetch_extract(title)
            if text and len(text) > 1000:
                dest.write_text(text, encoding="utf-8")
                print(f"  [ok]   {materia}/{dest.name}  ({len(text)} caratteri)")
            else:
                print(f"  [!!]   {materia}: voce '{title}' vuota o non trovata")
            time.sleep(2)  # cortesia verso l'API
    print("\nCorpus pronto in ./corpus — ora: python extract_lexicon.py")


if __name__ == "__main__":
    main()
