import csv
import json
import re
from pathlib import Path
import os

import requests
import spacy
import time


# ============================================================
# Configuration
# ============================================================

BASE = (
    "https://gitlab.lip6.fr/scialom/"
    "mlsum_data/-/raw/master/MLSUM"
)

FILES = {
    "train": f"{BASE}/fr_train.jsonl?inline=false",
}

USB_ROOT = Path(os.environ.get("QFR_CORPUS_ROOT", str(Path(__file__).resolve().parents[3] / "data" / "training_corpus" / "raw"))).expanduser().resolve()

if not USB_ROOT.exists():
    raise RuntimeError(
        f"Racine du corpus introuvable : {USB_ROOT}"
    )

OUT_DIR = USB_ROOT / "data" / "mlsum"
RAW_DIR = OUT_DIR / "raw"
PROCESSED_DIR = OUT_DIR / "processed"

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SpaCy
# ============================================================

nlp = spacy.load(
    "fr_core_news_sm",
    disable=[
        "ner",
        "tagger",
        "lemmatizer"
    ]
)

nlp.max_length = 3_000_000


# ============================================================
# Téléchargement
# ============================================================

MAX_DOWNLOAD_BYTES = 900_000_000  # 900 Mo


def download(url, path):
    """
    Télécharge volontairement environ 900 Mo de MLSUM.

    Le téléchargement s'arrête uniquement après une ligne JSONL
    complète afin que le fichier obtenu reste valide.
    """

    partial_path = Path(str(path) + ".part")

    # On supprime les restes d'anciens téléchargements incomplets
    if partial_path.exists():
        print(f"Suppression ancien fichier partiel : {partial_path}")
        partial_path.unlink()

    if path.exists():
        print(f"Fichier déjà présent : {path}")
        print(
            f"Taille : {path.stat().st_size / 1_000_000:.1f} Mo"
        )
        return

    print(f"Téléchargement : {url}")
    print(
        f"Limite volontaire : "
        f"{MAX_DOWNLOAD_BYTES / 1_000_000:.0f} Mo"
    )

    downloaded = 0
    article_count = 0

    with requests.get(
        url,
        stream=True,
        timeout=(30, 300),
    ) as r:

        r.raise_for_status()

        with partial_path.open("wb") as f:

            # On lit une ligne JSONL complète à la fois.
            for line in r.iter_lines(
                decode_unicode=False
            ):

                if not line:
                    continue

                line = line + b"\n"

                # Ne pas commencer un nouvel article
                # lorsque la limite est déjà atteinte.
                if (
                    downloaded >= MAX_DOWNLOAD_BYTES
                    and article_count > 0
                ):
                    break

                f.write(line)

                downloaded += len(line)
                article_count += 1

                if article_count % 5000 == 0:
                    print(
                        f"\r"
                        f"{downloaded / 1_000_000:.1f} Mo | "
                        f"{article_count:,} articles",
                        end="",
                        flush=True,
                    )

    print()

    # Le téléchargement volontairement partiel devient
    # notre fichier MLSUM exploitable.
    partial_path.rename(path)

    print("=" * 70)
    print("TÉLÉCHARGEMENT TERMINÉ")
    print("=" * 70)

    print(
        f"Taille : "
        f"{path.stat().st_size / 1_000_000:.1f} Mo"
    )

    print(
        f"Articles complets : "
        f"{article_count:,}"
    )

    print(
        f"Sauvegardé : {path}"
    )


# ============================================================
# Nettoyage
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = text.replace(
        "\xa0",
        " "
    )

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# Segmentation
# ============================================================

def sentence_split(text):
    if not text:
        return []

    doc = nlp(text)

    sentences = []

    for sent in doc.sents:

        s = sent.text.strip()

        if len(s) < 30:
            continue

        words = s.split()

        if len(words) < 5:
            continue

        if len(words) > 120:
            continue

        alpha_ratio = (
            sum(
                c.isalpha()
                for c in s
            )
            /
            max(
                len(s),
                1
            )
        )

        if alpha_ratio < 0.55:
            continue

        sentences.append(s)

    return sentences


# ============================================================
# Traitement d'un split
# ============================================================

def process_split(split):
    """
    Lit un JSONL MLSUM et écrit progressivement
    les phrases dans un CSV.
    """

    input_path = (
        RAW_DIR /
        f"fr_{split}.jsonl"
    )

    output_path = (
        PROCESSED_DIR /
        f"mlsum_{split}_sentences.csv"
    )

    print()
    print("=" * 70)
    print(
        f"TRAITEMENT : {split}"
    )
    print("=" * 70)

    fieldnames = [
        "text",
        "variety",
        "label",
        "genre",
        "source",
        "split_original",
        "article_id",
        "topic",
        "date",
        "url",
    ]

    article_count = 0
    sentence_count = 0
    error_count = 0

    with input_path.open(
        encoding="utf-8",
        errors="ignore"
    ) as f_in:

        with output_path.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as f_out:

            writer = csv.DictWriter(
                f_out,
                fieldnames=fieldnames
            )

            writer.writeheader()

            for article_id, line in enumerate(
                f_in
            ):

                try:

                    row = json.loads(
                        line
                    )

                    raw_text = row.get(
                        "text",
                        ""
                    )

                    article = clean_text(
                        raw_text
                    )

                    if not article:
                        continue

                    sentences = sentence_split(
                        article
                    )

                    for sentence in sentences:

                        writer.writerow({
                            "text": sentence,
                            "variety": "FI",
                            "label": 0,
                            "genre": "journalistic",
                            "source": "MLSUM",
                            "split_original": split,
                            "article_id": (
                                f"{split}_{article_id}"
                            ),
                            "topic": row.get(
                                "topic"
                            ),
                            "date": row.get(
                                "date"
                            ),
                            "url": row.get(
                                "url"
                            ),
                        })

                    sentence_count += len(
                        sentences
                    )

                    article_count += 1

                    if article_count % 1000 == 0:

                        print(
                            f"{article_count:,} articles | "
                            f"{sentence_count:,} phrases"
                        )

                except Exception as e:

                    error_count += 1

                    print(
                        f"Erreur "
                        f"{split}:{article_id}: "
                        f"{e}"
                    )

    print()
    print(
        f"Articles traités : "
        f"{article_count:,}"
    )

    print(
        f"Phrases sauvegardées : "
        f"{sentence_count:,}"
    )

    print(
        f"Erreurs : "
        f"{error_count:,}"
    )

    print(
        f"Sauvegardé : "
        f"{output_path}"
    )


# ============================================================
# Programme principal
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Télécharger les JSONL
    # --------------------------------------------------------

    for split, url in FILES.items():

        path = (
            RAW_DIR /
            f"fr_{split}.jsonl"
        )

        download(
            url,
            path
        )

    # --------------------------------------------------------
    # 2. Transformer en CSV par split
    # --------------------------------------------------------

    for split in FILES:

        process_split(
            split
        )

    print()
    print("=" * 70)
    print("TERMINÉ")
    print("=" * 70)

    print(
        f"Données brutes : "
        f"{RAW_DIR}"
    )

    print(
        f"Données traitées : "
        f"{PROCESSED_DIR}"
    )


if __name__ == "__main__":
    main()