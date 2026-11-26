import csv
import hashlib
import random
import re
import unicodedata
from pathlib import Path
import os


# ============================================================
# Configuration
# ============================================================

USB_ROOT = Path(os.environ.get("QFR_CORPUS_ROOT", str(Path(__file__).resolve().parents[3] / "data" / "training_corpus" / "raw"))).expanduser().resolve()
WIKI_DIR = USB_ROOT / "data" / "wiki"

FR_FILES = [
    WIKI_DIR / "wiki_fr_part_aa.txt",
    WIKI_DIR / "wiki_fr_part_ab.txt",
]

QC_FILE = (
    WIKI_DIR /
    "wikipedia_quebec_clean_v2.txt"
)

OUTPUT_DIR = (
    WIKI_DIR /
    "balanced"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUT_FR = (
    OUTPUT_DIR /
    "wikipedia_fr_balanced.csv"
)

OUT_QC = (
    OUTPUT_DIR /
    "wikipedia_qc_balanced.csv"
)

OVERLAP_OUT = (
    OUTPUT_DIR /
    "wikipedia_fr_qc_overlap.txt"
)

SEED = 42


# ============================================================
# Vérifications
# ============================================================

if not USB_ROOT.exists():
    raise RuntimeError(
        f"Racine du corpus introuvable : {USB_ROOT}"
    )

for path in FR_FILES + [QC_FILE]:

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )


# ============================================================
# Normalisation
# ============================================================

def normalize_text(text):
    """
    Normalisation uniquement pour comparer les textes.

    Le texte original est conservé dans les fichiers finaux.
    """

    text = text.strip()

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.lower()

    text = text.replace(
        "\xa0",
        " "
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = re.sub(
        r"\s+([.,;:!?])",
        r"\1",
        text
    )

    return text.strip()


# ============================================================
# Hash compact
# ============================================================

def text_hash(text):
    """
    Hash 64 bits utilisé pour conserver les index en mémoire
    sans stocker plusieurs millions de chaînes complètes.
    """

    return int.from_bytes(
        hashlib.blake2b(
            text.encode("utf-8"),
            digest_size=8
        ).digest(),
        byteorder="big"
    )


# ============================================================
# Charger Wikipedia QC
# ============================================================

def load_qc():
    """
    Charge le corpus QC.

    Comme QC est beaucoup plus petit que Wikipedia FR,
    il est raisonnable de le conserver en mémoire.
    """

    qc_texts = []
    qc_hashes = set()

    duplicates = 0
    empty = 0

    print("=" * 70)
    print("LECTURE WIKIPEDIA QC")
    print("=" * 70)

    with QC_FILE.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            original = line.strip()

            if not original:
                empty += 1
                continue

            normalized = normalize_text(
                original
            )

            if not normalized:
                empty += 1
                continue

            h = text_hash(
                normalized
            )

            if h in qc_hashes:
                duplicates += 1
                continue

            qc_hashes.add(h)
            qc_texts.append(original)

    print(
        f"Textes QC uniques : "
        f"{len(qc_texts):,}"
    )

    print(
        f"Doublons QC retirés : "
        f"{duplicates:,}"
    )

    print(
        f"Lignes vides ignorées : "
        f"{empty:,}"
    )

    return qc_texts, qc_hashes


# ============================================================
# Parcourir Wikipedia FR
# ============================================================

def process_fr(
    qc_hashes,
    target_size
):
    """
    Lit les deux morceaux Wikipedia FR comme un seul corpus.

    - retire les doublons FR internes
    - retire les textes également présents dans QC
    - effectue un reservoir sampling pour sélectionner
      exactement target_size exemples FR
    """

    rng = random.Random(
        SEED
    )

    # Hashes déjà rencontrés dans FR
    seen_fr = set()

    # Échantillon final
    reservoir = []

    total_lines = 0
    valid_fr = 0
    internal_duplicates = 0
    overlaps = 0
    empty = 0

    print()
    print("=" * 70)
    print("LECTURE WIKIPEDIA FR")
    print("=" * 70)

    with OVERLAP_OUT.open(
        "w",
        encoding="utf-8"
    ) as overlap_file:

        for fr_file in FR_FILES:

            print()
            print(
                f"Lecture : {fr_file.name}"
            )

            with fr_file.open(
                "r",
                encoding="utf-8",
                errors="ignore"
            ) as f:

                for line in f:

                    total_lines += 1

                    original = line.strip()

                    if not original:
                        empty += 1
                        continue

                    normalized = (
                        normalize_text(
                            original
                        )
                    )

                    if not normalized:
                        empty += 1
                        continue

                    h = text_hash(
                        normalized
                    )

                    # ----------------------------
                    # Doublon interne FR
                    # ----------------------------

                    if h in seen_fr:
                        internal_duplicates += 1
                        continue

                    seen_fr.add(h)

                    # ----------------------------
                    # Présent également dans QC
                    # ----------------------------

                    if h in qc_hashes:
                        overlaps += 1

                        overlap_file.write(
                            original + "\n"
                        )

                        continue

                    # ----------------------------
                    # Texte FI valide
                    # ----------------------------

                    valid_fr += 1

                    # Reservoir sampling
                    if len(reservoir) < target_size:

                        reservoir.append(
                            original
                        )

                    else:

                        j = rng.randint(
                            0,
                            valid_fr - 1
                        )

                        if j < target_size:
                            reservoir[j] = original

                    # Progression
                    if (
                        total_lines %
                        1_000_000
                        == 0
                    ):

                        print(
                            f"  {total_lines:,} lignes | "
                            f"{valid_fr:,} FI valides | "
                            f"{overlaps:,} chevauchements"
                        )

    print()
    print("=" * 70)
    print("STATISTIQUES FR")
    print("=" * 70)

    print(
        f"Lignes FR lues : "
        f"{total_lines:,}"
    )

    print(
        f"FR uniques hors QC : "
        f"{valid_fr:,}"
    )

    print(
        f"Doublons internes FR : "
        f"{internal_duplicates:,}"
    )

    print(
        f"Chevauchements FR/QC : "
        f"{overlaps:,}"
    )

    print(
        f"Lignes vides : "
        f"{empty:,}"
    )

    if valid_fr < target_size:

        raise RuntimeError(
            f"Pas assez de données FI : "
            f"{valid_fr:,} disponibles contre "
            f"{target_size:,} QC."
        )

    return (
        reservoir,
        {
            "total_lines": total_lines,
            "valid_fr": valid_fr,
            "internal_duplicates":
                internal_duplicates,
            "overlaps": overlaps,
        }
    )


# ============================================================
# Sauvegarder les corpus équilibrés
# ============================================================

def save_csv(
    texts,
    path,
    variety,
    label,
    source
):

    with path.open(
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "text",
                "variety",
                "label",
                "genre",
                "source",
            ]
        )

        writer.writeheader()

        for text in texts:

            writer.writerow({
                "text": text,
                "variety": variety,
                "label": label,
                "genre": "encyclopedic",
                "source": source,
            })


# ============================================================
# Main
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Québec
    # --------------------------------------------------------

    qc_texts, qc_hashes = (
        load_qc()
    )

    target_size = len(
        qc_texts
    )

    print()
    print(
        f"Objectif : {target_size:,} textes "
        f"par variété."
    )

    # --------------------------------------------------------
    # 2. France
    # --------------------------------------------------------

    fr_texts, stats = (
        process_fr(
            qc_hashes=qc_hashes,
            target_size=target_size
        )
    )

    # --------------------------------------------------------
    # 3. Mélanger QC avec seed fixe
    # --------------------------------------------------------

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        qc_texts
    )

    # --------------------------------------------------------
    # 4. Sauvegarde
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SAUVEGARDE")
    print("=" * 70)

    save_csv(
        texts=fr_texts,
        path=OUT_FR,
        variety="FI",
        label=0,
        source="Wikipedia_FR"
    )

    save_csv(
        texts=qc_texts,
        path=OUT_QC,
        variety="FQ",
        label=1,
        source="Wikipedia_QC"
    )

    # --------------------------------------------------------
    # 5. Résumé
    # --------------------------------------------------------

    n_overlap = (
        stats["overlaps"]
    )

    qc_n = len(
        qc_texts
    )

    fr_original = (
        stats["total_lines"]
    )

    overlap_qc_pct = (
        n_overlap /
        qc_n *
        100
        if qc_n
        else 0
    )

    overlap_fr_pct = (
        n_overlap /
        fr_original *
        100
        if fr_original
        else 0
    )

    print()
    print("=" * 70)
    print("CHEVAUCHEMENT FR / QC")
    print("=" * 70)

    print(
        f"Textes QC également trouvés "
        f"dans FR : {n_overlap:,}"
    )

    print(
        f"≈ {overlap_qc_pct:.2f}% "
        f"de la taille du corpus QC"
    )

    print(
        f"≈ {overlap_fr_pct:.4f}% "
        f"du corpus FR brut"
    )

    print()
    print("=" * 70)
    print("RÉSULTAT FINAL")
    print("=" * 70)

    print(
        f"FI : {len(fr_texts):,}"
    )

    print(
        f"FQ : {len(qc_texts):,}"
    )

    print(
        f"Total : "
        f"{len(fr_texts) + len(qc_texts):,}"
    )

    print()
    print(
        f"FR équilibré : {OUT_FR}"
    )

    print(
        f"QC équilibré : {OUT_QC}"
    )

    print(
        f"Chevauchements : "
        f"{OVERLAP_OUT}"
    )


if __name__ == "__main__":
    main()