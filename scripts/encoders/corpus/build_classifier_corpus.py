import json
import re
import unicodedata
from pathlib import Path
import os

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

USB_ROOT = Path(os.environ.get("QFR_CORPUS_ROOT", str(Path(__file__).resolve().parents[3] / "data" / "training_corpus" / "raw"))).expanduser().resolve()

OUTPUT_DIR = (
    USB_ROOT
    / "data"
    / "fq_fi_classifier_corpus"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CACHE_DIR = (
    OUTPUT_DIR
    / "genre_cache"
)

CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SEED = 42

# Maximum par variété et par genre
MAX_PER_VARIETY_PER_GENRE = 100_000

TRAIN_RATIO = 0.80
EVAL_RATIO = 0.10
TEST_RATIO = 0.10

# False = réutilise les caches déjà produits.
# True  = reconstruit complètement les 6 genres.
REBUILD_CACHE = os.environ.get("QFR_REUSE_GENRE_CACHE", "0") != "1"


# ============================================================
# CONFIGURATION DES CORPUS
# ============================================================
#
# types :
#
# csv
#     CSV avec colonne "text"
#
# txt_line
#     une unité textuelle par ligne
#
# txt_documents
#     lignes regroupées par document avec séparateur </s>
#
# txt_article_per_line
#     une ligne = un article/document
#
# vie_publique_json
#     JSON Vie-publique
#
# ============================================================

CORPORA = {

    # ========================================================
    # 1. Encyclopédique
    # ========================================================

    "encyclopedic": {

        "fi": [
            {
                "path":
                    USB_ROOT
                    / "data/wiki/balanced/"
                      "wikipedia_fr_balanced.csv",

                "type": "csv",

                "source": "Wikipedia_FR",
            }
        ],

        "fq": [
            {
                "path":
                    USB_ROOT
                    / "data/wiki/balanced/"
                      "wikipedia_qc_balanced.csv",

                "type": "csv",

                "source": "Wikipedia_QC",
            }
        ],
    },

    # ========================================================
    # 2. Parlementaire / institutionnel
    # ========================================================

    "parliamentary": {

        "fi": [
            {
                "path":
                    USB_ROOT
                    / "data/assemblee_fr/"
                      "assemblee_fr_sentences.csv",

                "type": "csv",

                "source": "Assemblee_FR",
            }
        ],

        "fq": [
            {
                "path":
                    USB_ROOT
                    / "data/assemblee_qc/"
                      "assemblee_qc_sentences.txt",

                "type": "txt_documents",

                "source": "Assemblee_QC",
            }
        ],
    },

    # ========================================================
    # 3. Rapports administratifs
    # ========================================================

    "administrative": {

        "fi": [
            {
                "path":
                    USB_ROOT
                    / "data/vie_publique/"
                      "vie_publique_sentences.json",

                "type": "vie_publique_json",

                "source": "Vie_publique",
            }
        ],

        "fq": [
            {
                "path":
                    USB_ROOT
                    / "data/vgq/"
                      "vgq_sentences.csv",

                "type": "csv",

                "source": "VGQ",
            }
        ],
    },

    # ========================================================
    # 4. Oral / conversationnel
    # ========================================================

    "oral": {

        "fi": [
            {
                "path":
                    USB_ROOT
                    / "data/orfeo/"
                      "orfeo_oral_sentences.csv",

                "type": "csv",

                "source": "ORFEO",
            }
        ],

        "fq": [
            {
                "path":
                    USB_ROOT
                    / "data/crifuq/"
                      "crifuq_sentences.txt",

                "type": "txt_documents",

                "source": "CRIFUQ",
            }
        ],
    },

    # ========================================================
    # 5. Informel Web
    # ========================================================

    "informal_web": {

        "fi": [
            {
                "path":
                    USB_ROOT
                    / "data/comere/"
                      "comere_informal_web_sentences.csv",

                "type": "csv",

                "source": "CoMeRe",
            }
        ],

        "fq": [
            {
                "path":
                    USB_ROOT
                    / "data/reddit/"
                      "reddit_qc_sentences.txt",

                "type": "txt_documents",

                "source": "Reddit_QC",
            },

            {
                "path":
                    USB_ROOT
                    / "data/forums/"
                      "forums_qc_sentences.txt",

                "type": "txt_line",

                "source": "Forums_QC",
            },
        ],
    },

    # ========================================================
    # 6. Journalistique
    # ========================================================

    "journalistic": {

        "fi": [
            {
                "path":
                    USB_ROOT
                    / "data/mlsum/processed/"
                      "mlsum_train_sentences.csv",

                "type": "csv",

                "source": "MLSUM",
            }
        ],

        "fq": [
            {
                "path":
                    USB_ROOT
                    / "data/cn2i/"
                      "cn2i_sentences.txt",

                "type": "txt_article_per_line",

                "source": "CN2i",
            },

            {
                "path":
                    USB_ROOT
                    / "data/radio_canada/"
                      "radio_canada_sentences.txt",

                "type": "txt_article_per_line",

                "source": "RadioCanada",
            },
        ],
    },
}


# ============================================================
# NORMALISATION
# ============================================================

def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKC",
        text
    )

    text = text.casefold()

    text = text.replace(
        "\xa0",
        " "
    )

    text = text.replace(
        "\u200b",
        " "
    )

    text = text.replace(
        "\ufeff",
        " "
    )

    text = text.replace(
        "</s>",
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
# FILTRAGE MINIMAL
# ============================================================

def valid_text(text):

    if text is None:
        return False

    text = str(text).strip()

    if not text:
        return False

    if len(text) < 10:
        return False

    words = text.split()

    if len(words) < 3:
        return False

    return True


# ============================================================
# CHARGEUR CSV
# ============================================================

def load_csv(path, source):

    print(
        f"Lecture CSV : {path}"
    )

    df = pd.read_csv(
        path,
        low_memory=False
    )

    if "text" not in df.columns:

        raise ValueError(
            f"{path} doit contenir "
            f"une colonne 'text'. "
            f"Colonnes disponibles : "
            f"{list(df.columns)}"
        )

    # --------------------------------------------------------
    # Identifier le document
    # --------------------------------------------------------

    candidates = [
        "document_id",
        "article_id",
        "document",
        "thread_id",
        "conversation_id",
        "session_id",
        "report_id",
        "publication_id",
        "speech_id",
        "debate_id",
        "url",
        "title",
    ]

    found = None

    for col in candidates:

        if col in df.columns:

            if df[col].notna().sum() > 0:
                found = col
                break

    if found:

        print(
            f"  document_id basé sur : "
            f"{found}"
        )

        df["document_id"] = (
            df[found]
            .fillna("")
            .astype(str)
        )

        # Remplacer les ID vides
        empty_mask = (
            df["document_id"]
            .str.strip()
            .eq("")
        )

        if empty_mask.any():

            df.loc[
                empty_mask,
                "document_id"
            ] = [
                f"{source}_row_{i}"
                for i in df.index[
                    empty_mask
                ]
            ]

    else:

        print(
            "  ATTENTION : aucune colonne "
            "document trouvée."
        )

        df["document_id"] = [
            f"{source}_row_{i}"
            for i in range(len(df))
        ]

    # Préfixer pour éviter les collisions
    df["document_id"] = (
        source
        + "::"
        + df["document_id"].astype(str)
    )

    df["source"] = source

    return df


# ============================================================
# TXT : UNE LIGNE = UNE UNITÉ
# ============================================================

def load_txt_lines(path, source):

    print(
        f"Lecture TXT ligne-par-ligne : "
        f"{path}"
    )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for i, line in enumerate(f):

            text = (
                line
                .replace("</s>", "")
                .strip()
            )

            if not valid_text(text):
                continue

            rows.append({
                "text": text,
                "document_id":
                    f"{source}::line_{i}",
                "source": source,
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# TXT : UNE LIGNE = UN ARTICLE / DOCUMENT
# ============================================================

def load_txt_article_per_line(
    path,
    source
):

    print(
        f"Lecture TXT article-par-ligne : "
        f"{path}"
    )

    rows = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for i, line in enumerate(f):

            text = (
                line
                .replace("</s>", "")
                .strip()
            )

            if not valid_text(text):
                continue

            rows.append({
                "text": text,
                "document_id":
                    f"{source}::article_{i}",
                "source": source,
            })

    return pd.DataFrame(
        rows
    )


# ============================================================
# TXT : DOCUMENTS SÉPARÉS PAR </s>
# ============================================================

def load_txt_documents(
    path,
    source
):

    print(
        f"Lecture TXT avec documents : "
        f"{path}"
    )

    rows = []

    document_id = 0

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            stripped = line.strip()

            if not stripped:
                continue

            # </s> peut être seul OU à la fin d'une ligne.
            ends_document = (
                "</s>" in stripped
            )

            text = (
                stripped
                .replace("</s>", "")
                .strip()
            )

            if valid_text(text):

                rows.append({
                    "text": text,
                    "document_id":
                        f"{source}::doc_{document_id}",
                    "source": source,
                })

            if ends_document:

                document_id += 1

    if rows:

        n_documents = (
            len({
                row["document_id"]
                for row in rows
            })
        )

    else:

        n_documents = 0

    print(
        f"  Documents détectés : "
        f"{n_documents:,}"
    )

    print(
        f"  Textes chargés : "
        f"{len(rows):,}"
    )

    return pd.DataFrame(
        rows
    )


# ============================================================
# VIE-PUBLIQUE JSON
# ============================================================

def load_vie_publique(
    path,
    source
):

    print(
        f"Lecture Vie-publique JSON : "
        f"{path}"
    )

    with path.open(
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    rows = []

    fields = [
        "titre",
        "presentation",
        "sommaire",
        "synthese",
        "extrait",
    ]

    for item in data:

        raw_id = item.get(
            "id"
        )

        if raw_id is None:
            continue

        document_id = (
            f"{source}::{raw_id}"
        )

        for field in fields:

            value = item.get(
                field
            )

            if not value:
                continue

            # Segmenter sur fin de phrase / retour ligne
            chunks = re.split(
                r"(?<=[.!?])\s+|\n+",
                str(value)
            )

            for text in chunks:

                text = text.strip()

                if not valid_text(text):
                    continue

                rows.append({
                    "text": text,
                    "document_id":
                        document_id,
                    "source": source,
                })

    return pd.DataFrame(
        rows
    )


# ============================================================
# CHARGEUR GÉNÉRAL
# ============================================================

def load_source(config):

    path = Path(
        config["path"]
    )

    source = (
        config["source"]
    )

    file_type = (
        config["type"]
    )

    if not path.exists():

        raise FileNotFoundError(
            f"Fichier introuvable : "
            f"{path}"
        )

    if file_type == "csv":

        return load_csv(
            path,
            source
        )

    if file_type == "txt_line":

        return load_txt_lines(
            path,
            source
        )

    if (
        file_type
        == "txt_article_per_line"
    ):

        return (
            load_txt_article_per_line(
                path,
                source
            )
        )

    if (
        file_type
        == "txt_documents"
    ):

        return load_txt_documents(
            path,
            source
        )

    if (
        file_type
        == "vie_publique_json"
    ):

        return load_vie_publique(
            path,
            source
        )

    raise ValueError(
        f"Type inconnu : "
        f"{file_type}"
    )


# ============================================================
# CHARGER UNE VARIÉTÉ
# ============================================================

def load_variety(
    source_configs,
    variety,
    genre
):

    dfs = []

    for config in source_configs:

        df = load_source(
            config
        )

        if df.empty:

            print(
                f"ATTENTION : "
                f"{config['source']} est vide."
            )

            continue

        dfs.append(
            df
        )

    if not dfs:

        raise RuntimeError(
            f"Aucune donnée disponible "
            f"pour {genre}/{variety}"
        )

    df = pd.concat(
        dfs,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Nettoyage
    # --------------------------------------------------------

    df["text"] = (
        df["text"]
        .astype(str)
        .str.strip()
    )

    df = df[
        df["text"]
        .map(valid_text)
    ].copy()

    df["_normalized"] = (
        df["text"]
        .map(normalize_text)
    )

    df = df[
        df["_normalized"]
        .str.len()
        > 0
    ].copy()

    # --------------------------------------------------------
    # Déduplication interne à la variété/genre
    # --------------------------------------------------------

    before = len(df)

    df = (
        df.drop_duplicates(
            subset=[
                "_normalized"
            ],
            keep="first"
        )
        .reset_index(
            drop=True
        )
    )

    print(
        f"Doublons internes supprimés : "
        f"{before - len(df):,}"
    )

    # --------------------------------------------------------
    # Métadonnées
    # --------------------------------------------------------

    df["variety"] = variety

    df["label"] = (
        0
        if variety == "FI"
        else 1
    )

    df["genre"] = genre

    return df


# ============================================================
# RETIRER CHEVAUCHEMENT FI/FQ DANS UN GENRE
# ============================================================

def remove_cross_overlap(
    fi,
    fq
):

    fi_set = set(
        fi["_normalized"]
    )

    fq_set = set(
        fq["_normalized"]
    )

    overlap = (
        fi_set
        & fq_set
    )

    print(
        f"Chevauchements FI/FQ : "
        f"{len(overlap):,}"
    )

    if overlap:

        fi = fi[
            ~fi["_normalized"]
            .isin(overlap)
        ].copy()

        fq = fq[
            ~fq["_normalized"]
            .isin(overlap)
        ].copy()

    return (
        fi.reset_index(
            drop=True
        ),
        fq.reset_index(
            drop=True
        )
    )


# ============================================================
# SPLIT PAR DOCUMENT
# ============================================================

def split_documents(
    df,
    seed
):

    docs = (
        df.groupby(
            "document_id"
        )
        .size()
        .reset_index(
            name="n"
        )
    )

    if len(docs) < 3:

        raise RuntimeError(
            "Impossible de construire "
            "train/eval/test : moins de "
            "3 documents."
        )

    docs = (
        docs.sample(
            frac=1,
            random_state=seed
        )
        .reset_index(
            drop=True
        )
    )

    total = (
        docs["n"].sum()
    )

    targets = {
        "train":
            total * TRAIN_RATIO,

        "eval":
            total * EVAL_RATIO,

        "test":
            total * TEST_RATIO,
    }

    current = {
        "train": 0,
        "eval": 0,
        "test": 0,
    }

    assignments = {}

    for _, row in docs.iterrows():

        doc = (
            row["document_id"]
        )

        n = int(
            row["n"]
        )

        deficits = {}

        for split in targets:

            deficits[split] = (
                (
                    targets[split]
                    - current[split]
                )
                /
                max(
                    targets[split],
                    1
                )
            )

        chosen = max(
            deficits,
            key=deficits.get
        )

        assignments[
            doc
        ] = chosen

        current[
            chosen
        ] += n

    result = df.copy()

    result["split"] = (
        result["document_id"]
        .map(assignments)
    )

    return result


# ============================================================
# CONSTRUCTION D'UN GENRE
# ============================================================

def build_genre(
    genre,
    config
):

    print()
    print("=" * 80)
    print(
        genre.upper()
    )
    print("=" * 80)

    # --------------------------------------------------------
    # FI
    # --------------------------------------------------------

    print()
    print("FI")

    fi = load_variety(
        config["fi"],
        "FI",
        genre
    )

    # --------------------------------------------------------
    # FQ
    # --------------------------------------------------------

    print()
    print("FQ")

    fq = load_variety(
        config["fq"],
        "FQ",
        genre
    )

    print()

    print(
        f"FI disponible : "
        f"{len(fi):,}"
    )

    print(
        f"FQ disponible : "
        f"{len(fq):,}"
    )

    # --------------------------------------------------------
    # Retirer textes identiques FI/FQ
    # --------------------------------------------------------

    fi, fq = (
        remove_cross_overlap(
            fi,
            fq
        )
    )

    # --------------------------------------------------------
    # Split par document AVANT échantillonnage
    # --------------------------------------------------------

    fi = split_documents(
        fi,
        SEED
    )

    fq = split_documents(
        fq,
        SEED + 1000
    )

    # --------------------------------------------------------
    # Taille cible par variété
    # --------------------------------------------------------

    total_target = min(
        len(fi),
        len(fq),
        MAX_PER_VARIETY_PER_GENRE
    )

    targets = {
        "train":
            int(
                total_target
                * TRAIN_RATIO
            ),

        "eval":
            int(
                total_target
                * EVAL_RATIO
            ),
    }

    targets["test"] = (
        total_target
        - targets["train"]
        - targets["eval"]
    )

    print()

    print(
        f"Cible totale/variété : "
        f"{total_target:,}"
    )

    final = []

    split_seed = {
        "train": 1,
        "eval": 2,
        "test": 3,
    }

    # --------------------------------------------------------
    # Équilibrage strict FI/FQ dans chaque split
    # --------------------------------------------------------

    for split in [
        "train",
        "eval",
        "test"
    ]:

        fi_s = fi[
            fi["split"] == split
        ].copy()

        fq_s = fq[
            fq["split"] == split
        ].copy()

        n = min(
            len(fi_s),
            len(fq_s),
            targets[split]
        )

        if n == 0:

            raise RuntimeError(
                f"{genre}/{split} est vide. "
                f"FI={len(fi_s):,}, "
                f"FQ={len(fq_s):,}"
            )

        fi_s = (
            fi_s.sample(
                n=n,
                random_state=(
                    SEED
                    + split_seed[split]
                )
            )
            .copy()
        )

        fq_s = (
            fq_s.sample(
                n=n,
                random_state=(
                    SEED
                    + 100
                    + split_seed[split]
                )
            )
            .copy()
        )

        final.extend([
            fi_s,
            fq_s
        ])

        print(
            f"{split}: "
            f"{n:,} FI + "
            f"{n:,} FQ"
        )

    return pd.concat(
        final,
        ignore_index=True
    )


# ============================================================
# NETTOYAGE GLOBAL
# ============================================================

def remove_global_cross_split_duplicates(
    df
):
    """
    Supprime entièrement tout texte normalisé qui apparaît
    dans plusieurs splits.

    Exemple :
        même phrase dans train et test
        -> suppression de toutes les occurrences.

    Cela garantit qu'aucun texte identique ne fuit entre
    train/eval/test.
    """

    counts = (
        df.groupby(
            "_normalized"
        )["split"]
        .nunique()
    )

    leaking_texts = set(
        counts[
            counts > 1
        ].index
    )

    print()
    print("=" * 80)
    print(
        "NETTOYAGE GLOBAL "
        "DES FUITES TEXTUELLES"
    )
    print("=" * 80)

    print(
        f"Textes présents dans "
        f"plusieurs splits : "
        f"{len(leaking_texts):,}"
    )

    if not leaking_texts:

        print(
            "Aucune fuite textuelle "
            "à supprimer."
        )

        return df.reset_index(
            drop=True
        )

    before = len(df)

    df = df[
        ~df["_normalized"]
        .isin(leaking_texts)
    ].copy()

    removed = (
        before - len(df)
    )

    print(
        f"Lignes supprimées : "
        f"{removed:,}"
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# CONFLITS GLOBAUX FI / FQ
# ============================================================

def remove_global_label_conflicts(
    df
):
    """
    Retire tout texte qui possède les deux labels
    quelque part dans le corpus.

    Exemple :
        "..." label=0 dans une source
        "..." label=1 dans une autre
        -> les deux sont supprimés.
    """

    label_counts = (
        df.groupby(
            "_normalized"
        )["label"]
        .nunique()
    )

    conflicting = set(
        label_counts[
            label_counts > 1
        ].index
    )

    print()
    print(
        f"Textes avec conflit "
        f"FI/FQ global : "
        f"{len(conflicting):,}"
    )

    if not conflicting:

        return df.reset_index(
            drop=True
        )

    before = len(df)

    df = df[
        ~df["_normalized"]
        .isin(conflicting)
    ].copy()

    print(
        f"Lignes supprimées pour "
        f"conflits de labels : "
        f"{before - len(df):,}"
    )

    return df.reset_index(
        drop=True
    )


# ============================================================
# RÉÉQUILIBRAGE APRÈS NETTOYAGE
# ============================================================

def rebalance_after_cleanup(
    df,
    seed=42
):
    """
    Après retrait des fuites globales, conserve
    exactement autant de FI que de FQ pour chaque :

        genre × split
    """

    print()
    print("=" * 80)
    print(
        "RÉÉQUILIBRAGE APRÈS NETTOYAGE"
    )
    print("=" * 80)

    parts = []

    groups = (
        df.groupby(
            [
                "genre",
                "split"
            ],
            sort=False
        )
    )

    for (
        genre,
        split
    ), group in groups:

        fi = group[
            group["label"] == 0
        ].copy()

        fq = group[
            group["label"] == 1
        ].copy()

        n = min(
            len(fi),
            len(fq)
        )

        if n == 0:

            raise RuntimeError(
                f"Split vide après "
                f"nettoyage : "
                f"{genre}/{split}. "
                f"FI={len(fi):,}, "
                f"FQ={len(fq):,}"
            )

        fi = fi.sample(
            n=n,
            random_state=seed
        )

        fq = fq.sample(
            n=n,
            random_state=seed + 1
        )

        parts.extend([
            fi,
            fq
        ])

        print(
            f"{genre:18s} "
            f"{split:6s} : "
            f"{n:,} FI + "
            f"{n:,} FQ"
        )

    result = pd.concat(
        parts,
        ignore_index=True
    )

    return result


# ============================================================
# VÉRIFICATIONS
# ============================================================

def verify_no_document_leakage(
    df
):

    counts = (
        df.groupby(
            [
                "genre",
                "variety",
                "document_id"
            ]
        )["split"]
        .nunique()
    )

    leaks = counts[
        counts > 1
    ]

    if len(leaks):

        raise RuntimeError(
            f"{len(leaks):,} documents "
            f"présents dans plusieurs splits."
        )

    print(
        "✓ Aucun document partagé "
        "entre train/eval/test"
    )


def verify_no_text_leakage(
    df
):

    counts = (
        df.groupby(
            "_normalized"
        )["split"]
        .nunique()
    )

    leaks = counts[
        counts > 1
    ]

    if len(leaks):

        raise RuntimeError(
            f"{len(leaks):,} textes "
            f"présents dans plusieurs splits."
        )

    print(
        "✓ Aucun texte partagé "
        "entre train/eval/test"
    )


def verify_no_label_conflicts(
    df
):

    counts = (
        df.groupby(
            "_normalized"
        )["label"]
        .nunique()
    )

    conflicts = counts[
        counts > 1
    ]

    if len(conflicts):

        raise RuntimeError(
            f"{len(conflicts):,} textes "
            f"possèdent à la fois "
            f"les labels FI et FQ."
        )

    print(
        "✓ Aucun texte avec "
        "labels FI/FQ contradictoires"
    )


def verify_balance(
    df
):

    problems = []

    grouped = (
        df.groupby(
            [
                "genre",
                "split",
                "label"
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
    )

    for index, row in (
        grouped.iterrows()
    ):

        genre, split = index

        fi = int(
            row.get(
                0,
                0
            )
        )

        fq = int(
            row.get(
                1,
                0
            )
        )

        if fi != fq:

            problems.append(
                (
                    genre,
                    split,
                    fi,
                    fq
                )
            )

    if problems:

        raise RuntimeError(
            "Déséquilibre FI/FQ détecté : "
            f"{problems}"
        )

    print(
        "✓ FI = FQ dans chaque "
        "genre et chaque split"
    )


# ============================================================
# STATISTIQUES
# ============================================================

def build_statistics(
    df
):

    stats = (
        df.groupby(
            [
                "split",
                "genre",
                "variety",
                "label"
            ]
        )
        .size()
        .reset_index(
            name="n"
        )
    )

    return stats


# ============================================================
# SAUVEGARDE
# ============================================================

def save_final_splits(
    df
):

    split_files = {}

    for split in [
        "train",
        "eval",
        "test"
    ]:

        out = df[
            df["split"] == split
        ].copy()

        out = (
            out.sample(
                frac=1,
                random_state=SEED
            )
            .reset_index(
                drop=True
            )
        )

        # Colonne temporaire seulement utilisée
        # pour nettoyage / vérifications.
        out = out.drop(
            columns=[
                "_normalized"
            ],
            errors="ignore"
        )

        output_path = (
            OUTPUT_DIR
            / f"{split}.csv"
        )

        out.to_csv(
            output_path,
            index=False
        )

        split_files[
            split
        ] = out

        print()
        print(
            f"{split.upper()} : "
            f"{len(out):,}"
        )

        print(
            f"  FI : "
            f"{(out['label'] == 0).sum():,}"
        )

        print(
            f"  FQ : "
            f"{(out['label'] == 1).sum():,}"
        )

        print(
            f"  -> {output_path}"
        )

    return split_files


# ============================================================
# MAIN
# ============================================================

def main():

    parts = []

    # ========================================================
    # 1. Construire ou charger chaque genre
    # ========================================================

    for genre, config in (
        CORPORA.items()
    ):

        cache_path = (
            CACHE_DIR
            / f"{genre}.csv"
        )

        if (
            cache_path.exists()
            and not REBUILD_CACHE
        ):

            print()
            print("=" * 80)
            print(
                f"{genre.upper()} : "
                f"résultat déjà disponible"
            )
            print("=" * 80)

            print(
                f"Chargement : "
                f"{cache_path}"
            )

            df_genre = pd.read_csv(
                cache_path,
                low_memory=False
            )

            required_cache_columns = {
                "text",
                "document_id",
                "source",
                "_normalized",
                "variety",
                "label",
                "genre",
                "split",
            }

            missing = (
                required_cache_columns
                - set(
                    df_genre.columns
                )
            )

            if missing:

                raise RuntimeError(
                    f"Cache invalide : "
                    f"{cache_path}\n"
                    f"Colonnes manquantes : "
                    f"{sorted(missing)}\n"
                    f"Mettre REBUILD_CACHE=True."
                )

        else:

            df_genre = build_genre(
                genre,
                config
            )

            df_genre.to_csv(
                cache_path,
                index=False
            )

            print(
                f"Sauvegarde intermédiaire : "
                f"{cache_path}"
            )

        parts.append(
            df_genre
        )

    # ========================================================
    # 2. Fusion globale
    # ========================================================

    df = pd.concat(
        parts,
        ignore_index=True
    )

    print()
    print("=" * 80)
    print("CORPUS AVANT NETTOYAGE GLOBAL")
    print("=" * 80)

    print(
        f"Total : "
        f"{len(df):,}"
    )

    print(
        f"FI : "
        f"{(df['label'] == 0).sum():,}"
    )

    print(
        f"FQ : "
        f"{(df['label'] == 1).sum():,}"
    )

    # ========================================================
    # 3. Supprimer conflits FI/FQ globaux
    # ========================================================

    df = remove_global_label_conflicts(
        df
    )

    # ========================================================
    # 4. Supprimer textes apparaissant dans plusieurs splits
    # ========================================================

    df = (
        remove_global_cross_split_duplicates(
            df
        )
    )

    # ========================================================
    # 5. Rééquilibrage après nettoyage
    # ========================================================

    df = rebalance_after_cleanup(
        df,
        seed=SEED
    )

    # ========================================================
    # 6. Vérifications finales
    # ========================================================

    print()
    print("=" * 80)
    print("VÉRIFICATIONS FINALES")
    print("=" * 80)

    verify_no_document_leakage(
        df
    )

    verify_no_text_leakage(
        df
    )

    verify_no_label_conflicts(
        df
    )

    verify_balance(
        df
    )

    # ========================================================
    # 7. Statistiques
    # ========================================================

    stats = build_statistics(
        df
    )

    print()
    print("=" * 80)
    print("RÉPARTITION FINALE")
    print("=" * 80)

    print(
        stats.to_string(
            index=False
        )
    )

    stats_path = (
        OUTPUT_DIR
        / "corpus_statistics.csv"
    )

    stats.to_csv(
        stats_path,
        index=False
    )

    # ========================================================
    # 8. Sauvegarder train / eval / test
    # ========================================================

    split_files = (
        save_final_splits(
            df
        )
    )

    # ========================================================
    # 9. Corpus complet
    # ========================================================

    complete = pd.concat(
        [
            split_files["train"],
            split_files["eval"],
            split_files["test"],
        ],
        ignore_index=True
    )

    complete_path = (
        OUTPUT_DIR
        / "fq_fi_classifier_all.csv"
    )

    complete.to_csv(
        complete_path,
        index=False
    )

    # ========================================================
    # 10. Résumé
    # ========================================================

    print()
    print("=" * 80)
    print("TERMINÉ")
    print("=" * 80)

    print(
        f"Train : "
        f"{len(split_files['train']):,}"
    )

    print(
        f"Eval  : "
        f"{len(split_files['eval']):,}"
    )

    print(
        f"Test  : "
        f"{len(split_files['test']):,}"
    )

    print(
        f"Total : "
        f"{len(complete):,}"
    )

    print()

    print(
        "Labels globaux :"
    )

    print(
        complete[
            "variety"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        f"Statistiques : "
        f"{stats_path}"
    )

    print(
        f"Corpus complet : "
        f"{complete_path}"
    )

    print(
        f"Dossier final : "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()