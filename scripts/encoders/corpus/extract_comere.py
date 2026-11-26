import csv
import re
from pathlib import Path
import os

import spacy
from lxml import etree


# ============================================================
# Configuration
# ============================================================

USB_ROOT = Path(os.environ.get("QFR_CORPUS_ROOT", str(Path(__file__).resolve().parents[3] / "data" / "training_corpus" / "raw"))).expanduser().resolve()

if not USB_ROOT.exists():
    raise RuntimeError(
        f"Racine du corpus introuvable : {USB_ROOT}"
    )

OUT_DIR = USB_ROOT / "data" / "comere"
EXTRACT_DIR = OUT_DIR / "extracted"

OUTPUT_CSV = (
    OUT_DIR /
    "comere_informal_web_sentences.csv"
)

# Nombre maximum total de phrases FI
MAX_TOTAL_SENTENCES = 200_000

# Nombre maximum provenant d'un même sous-corpus
MAX_PER_SUBCORPUS = 30_000


# ============================================================
# Sous-corpus à exclure
# ============================================================

EXCLUDED_PATTERNS = {
    "smslareunion",
    "88milsms",
}

EXCLUDED_PREFIXES = {
    "olac-",
    "._",
}


# ============================================================
# SpaCy
# ============================================================

nlp = spacy.load(
    "fr_core_news_sm",
    disable=[
        "ner",
        "tagger",
        "lemmatizer",
    ],
)

nlp.max_length = 3_000_000


# ============================================================
# Vérification fichiers
# ============================================================

def should_keep_file(path):
    filename = path.name.lower()
    full_path = str(path).lower()

    # Fichiers macOS parasites
    if filename.startswith("._"):
        return False

    if filename == ".ds_store":
        return False

    # Fichiers OLAC / métadonnées
    for prefix in EXCLUDED_PREFIXES:
        if filename.startswith(prefix):
            return False

    # Sous-corpus exclus
    for pattern in EXCLUDED_PATTERNS:
        if pattern in full_path:
            return False

    # Évite les fichiers vides
    try:
        if path.stat().st_size == 0:
            return False
    except OSError:
        return False

    return True


# ============================================================
# Nettoyage
# ============================================================

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", " ")
    text = text.replace("\ufeff", " ")

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
    )

    text = re.sub(
        r"\bwww\.\S+",
        " ",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# ============================================================
# XML / TEI
# ============================================================

def extract_xml_text(path):
    parser = etree.XMLParser(
        recover=True,
        huge_tree=True,
        remove_comments=True,
    )

    tree = etree.parse(
        str(path),
        parser,
    )

    root = tree.getroot()

    blocks = []

    interesting_tags = {
        "u",
        "p",
        "s",
        "post",
        "message",
        "utterance",
        "seg",
        "ab",
    }

    for element in root.iter():

        try:
            tag = etree.QName(
                element
            ).localname.lower()

        except Exception:
            continue

        if tag not in interesting_tags:
            continue

        text = " ".join(
            part.strip()
            for part in element.itertext()
            if part and part.strip()
        )

        text = clean_text(text)

        if len(text) >= 5:
            blocks.append(text)

    if not blocks:
        text = " ".join(
            part.strip()
            for part in root.itertext()
            if part and part.strip()
        )

        text = clean_text(text)

        if text:
            blocks.append(text)

    return blocks


# ============================================================
# TXT
# ============================================================

def extract_txt(path):
    encodings = [
        "utf-8",
        "utf-8-sig",
        "latin-1",
        "cp1252",
    ]

    for encoding in encodings:

        try:
            text = path.read_text(
                encoding=encoding
            )

            return [
                clean_text(text)
            ]

        except UnicodeDecodeError:
            continue

    text = path.read_text(
        encoding="utf-8",
        errors="ignore",
    )

    return [
        clean_text(text)
    ]


# ============================================================
# Segmentation en phrases
# ============================================================

def sentence_split(text):
    if not text:
        return []

    chunk_size = 500_000

    chunks = [
        text[i:i + chunk_size]
        for i in range(
            0,
            len(text),
            chunk_size,
        )
    ]

    results = []

    for chunk in chunks:

        doc = nlp(chunk)

        for sent in doc.sents:

            sentence = sent.text.strip()

            if not sentence:
                continue

            words = sentence.split()

            if len(sentence) < 10:
                continue

            if len(words) < 3:
                continue

            if len(words) > 120:
                continue

            alpha_count = sum(
                char.isalpha()
                for char in sentence
            )

            alpha_ratio = (
                alpha_count /
                max(
                    len(sentence),
                    1
                )
            )

            if alpha_ratio < 0.40:
                continue

            results.append(
                sentence
            )

    return results


# ============================================================
# Sous-corpus
# ============================================================

def get_subcorpus_name(path):
    for part in path.parts:

        part_lower = part.lower()

        if part_lower.startswith(
            "cmr-"
        ):
            return part_lower

    return "unknown"


# ============================================================
# Type Web
# ============================================================

def detect_type(path):
    joined = "/".join(
        part.lower()
        for part in path.parts
    )

    keywords = {
        "forum": "forum",
        "blog": "blog",
        "chat": "chat",
        "clavard": "chat",
        "wiki": "wiki",
        "mail": "email",
        "email": "email",
        "tweet": "social_media",
        "twitter": "social_media",
        "facebook": "social_media",
        "comment": "comments",
    }

    for keyword, category in keywords.items():

        if keyword in joined:
            return category

    return "unknown"


# ============================================================
# Fichiers déjà extraits
# ============================================================

def get_existing_files():
    files = []

    for suffix in [
        "*.xml",
        "*.tei",
        "*.txt",
    ]:
        files.extend(
            EXTRACT_DIR.rglob(
                suffix
            )
        )

    files = [
        path
        for path in files
        if should_keep_file(path)
    ]

    return sorted(files)


# ============================================================
# Programme principal
# ============================================================

def main():

    files = get_existing_files()

    print(
        f"{len(files):,} fichiers utiles trouvés "
        f"dans {EXTRACT_DIR}"
    )

    if not files:
        raise RuntimeError(
            "Aucun fichier CoMeRe extrait trouvé."
        )

    fieldnames = [
        "text",
        "variety",
        "label",
        "genre",
        "source",
        "subcorpus",
        "web_type",
        "document",
    ]

    total_count = 0

    subcorpus_counts = {}

    seen_texts = set()

    with OUTPUT_CSV.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f_out:

        writer = csv.DictWriter(
            f_out,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for i, path in enumerate(
            files,
            start=1
        ):

            if total_count >= MAX_TOTAL_SENTENCES:
                break

            relative_path = (
                path.relative_to(
                    EXTRACT_DIR
                )
            )

            subcorpus = (
                get_subcorpus_name(
                    relative_path
                )
            )

            current_subcorpus_count = (
                subcorpus_counts.get(
                    subcorpus,
                    0
                )
            )

            # Si ce sous-corpus a déjà atteint son plafond
            if (
                current_subcorpus_count
                >= MAX_PER_SUBCORPUS
            ):
                continue

            print(
                f"[{i}/{len(files)}] "
                f"{relative_path}"
            )

            try:

                suffix = (
                    path
                    .suffix
                    .lower()
                )

                if suffix in {
                    ".xml",
                    ".tei",
                }:
                    blocks = extract_xml_text(
                        path
                    )

                elif suffix == ".txt":
                    blocks = extract_txt(
                        path
                    )

                else:
                    continue

                file_count = 0

                stop_subcorpus = False

                for block in blocks:

                    sentences = sentence_split(
                        block
                    )

                    for sentence in sentences:

                        if (
                            total_count
                            >= MAX_TOTAL_SENTENCES
                        ):
                            break

                        if (
                            subcorpus_counts.get(
                                subcorpus,
                                0
                            )
                            >= MAX_PER_SUBCORPUS
                        ):
                            stop_subcorpus = True
                            break

                        # Déduplication globale
                        normalized = (
                            sentence
                            .strip()
                            .lower()
                        )

                        if normalized in seen_texts:
                            continue

                        seen_texts.add(
                            normalized
                        )

                        writer.writerow({
                            "text": sentence,
                            "variety": "FI",
                            "label": 0,
                            "genre": "informal_web",
                            "source": "CoMeRe",
                            "subcorpus": subcorpus,
                            "web_type": detect_type(
                                relative_path
                            ),
                            "document": str(
                                relative_path
                            ),
                        })

                        total_count += 1

                        file_count += 1

                        subcorpus_counts[
                            subcorpus
                        ] = (
                            subcorpus_counts.get(
                                subcorpus,
                                0
                            )
                            + 1
                        )

                    if (
                        total_count
                        >= MAX_TOTAL_SENTENCES
                    ):
                        break

                    if stop_subcorpus:
                        break

                print(
                    f"  -> {file_count:,} phrases gardées"
                )

                print(
                    f"  Total : {total_count:,} / "
                    f"{MAX_TOTAL_SENTENCES:,}"
                )

                print(
                    f"  {subcorpus} : "
                    f"{subcorpus_counts.get(subcorpus, 0):,} / "
                    f"{MAX_PER_SUBCORPUS:,}"
                )

            except Exception as e:

                print(
                    f"  ERREUR : {e}"
                )

    # ========================================================
    # Résumé
    # ========================================================

    print()
    print("=" * 70)
    print("TERMINÉ")
    print("=" * 70)

    print(
        f"Total final : "
        f"{total_count:,} phrases"
    )

    print()
    print(
        "Répartition par sous-corpus :"
    )

    for subcorpus, count in sorted(
        subcorpus_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{subcorpus}: {count:,}"
        )

    print()
    print(
        f"Sauvegardé : "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()