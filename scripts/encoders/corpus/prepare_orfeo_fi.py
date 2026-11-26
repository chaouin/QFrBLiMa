from pathlib import Path
import argparse
import csv
import random
import re


# ------------------------------------------------------------------
# Corpus sélectionnés
# ------------------------------------------------------------------

KEEP_WRITTEN = {
    "comere",
}

KEEP_ORAL = {
    "cfpp",
    "crfp",
    "clapi",
}

EXCLUDE = {
    "ofrom",       # Suisse romande
    "valibel",     # Belgique
}


# ------------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------------

def normalize(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\u00a0", " ")
    text = text.replace("\u202f", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def corpus_path_string(path: Path) -> str:
    """
    Retourne le chemin complet en minuscules pour identifier
    le corpus à partir du nom du dossier.
    """
    return "/".join(part.lower() for part in path.parts)


# ------------------------------------------------------------------
# Sélection des fichiers
# ------------------------------------------------------------------

def wanted_file(path: Path) -> bool:
    """
    Ne conserve que les fichiers .orfeo appartenant aux corpus
    sélectionnés.
    """
    if path.suffix.lower() != ".orfeo":
        return False

    name = corpus_path_string(path)

    if any(x in name for x in EXCLUDE):
        return False

    return (
        any(x in name for x in KEEP_WRITTEN)
        or any(x in name for x in KEEP_ORAL)
    )


def identify_corpus(path: Path) -> str:
    """
    Identifie précisément le corpus source.
    """
    name = corpus_path_string(path)

    if "comere" in name:
        return "CoMeRe"

    if "cfpp" in name:
        return "CFPP"

    if "crfp" in name:
        return "CRFP"

    if "clapi" in name:
        return "CLAPI"

    return "unknown"


# ------------------------------------------------------------------
# Reconstruction du texte
# ------------------------------------------------------------------

def detokenize(tokens):
    """
    Reconstruit une phrase lisible à partir des FORM ORFEO.
    """
    if not tokens:
        return ""

    text = " ".join(tokens)

    # Ponctuation simple
    text = re.sub(
        r"\s+([.,!?…])",
        r"\1",
        text
    )

    # Ponctuation française
    text = re.sub(
        r"\s+([;:])",
        r"\1",
        text
    )

    # Parenthèses
    text = re.sub(
        r"\(\s+",
        "(",
        text
    )

    text = re.sub(
        r"\s+\)",
        ")",
        text
    )

    # Guillemets
    text = re.sub(
        r"«\s+",
        "« ",
        text
    )

    text = re.sub(
        r"\s+»",
        " »",
        text
    )

    # Apostrophes françaises :
    # l' homme -> l'homme
    # j' ai -> j'ai
    # qu' il -> qu'il
    text = re.sub(
        r"\b([cdjlmnstqu])['’]\s+",
        r"\1'",
        text,
        flags=re.IGNORECASE
    )

    return normalize(text)


# ------------------------------------------------------------------
# Lecture ORFEO
# ------------------------------------------------------------------

def read_orfeo(path: Path):
    """
    Lit un fichier .orfeo.

    Format attendu :
        colonne 1 = ID
        colonne 2 = FORM

    Les lignes vides sont utilisées comme séparateurs de phrases
    ou d'unités annotées.
    """

    try:
        with path.open(
            "r",
            encoding="utf-8",
            errors="ignore"
        ) as f:

            tokens = []

            for raw_line in f:

                line = raw_line.rstrip("\n\r")

                # --------------------------------------------------
                # Ligne vide = fin d'une unité / phrase
                # --------------------------------------------------

                if not line.strip():

                    if tokens:
                        sentence = detokenize(tokens)

                        if sentence:
                            yield sentence

                        tokens = []

                    continue

                # --------------------------------------------------
                # Commentaires éventuels
                # --------------------------------------------------

                if line.startswith("#"):
                    continue

                # --------------------------------------------------
                # Ligne TSV ORFEO
                # --------------------------------------------------

                fields = line.split("\t")

                if len(fields) < 2:
                    continue

                token_id = fields[0].strip()
                form = fields[1].strip()

                # IDs normaux : 1, 2, 3...
                if not token_id.isdigit():
                    continue

                if not form:
                    continue

                if form == "_":
                    continue

                tokens.append(form)

            # Dernière phrase si absence de ligne vide finale
            if tokens:

                sentence = detokenize(tokens)

                if sentence:
                    yield sentence

    except Exception as e:

        print(
            f"[WARN] Impossible de lire {path}: {e}"
        )


# ------------------------------------------------------------------
# Validation des phrases
# ------------------------------------------------------------------

def valid_sentence(text: str) -> bool:

    text = normalize(text)

    if not text:
        return False

    words = text.split()

    # Évite fragments extrêmement courts
    if len(words) < 5:
        return False

    # Évite les unités anormalement longues
    if len(words) > 80:
        return False

    # Doit contenir du texte alphabétique
    if not re.search(
        r"[A-Za-zÀ-ÿ]",
        text
    ):
        return False

    # Évite les lignes principalement constituées
    # de symboles ou d'annotations
    alpha_chars = len(
        re.findall(
            r"[A-Za-zÀ-ÿ]",
            text
        )
    )

    if (
        alpha_chars
        / max(len(text), 1)
        < 0.45
    ):
        return False

    return True


# ------------------------------------------------------------------
# Extraction
# ------------------------------------------------------------------

def extract(root: Path):

    data = {
        "written": [],
        "oral": [],
    }

    counts_by_corpus = {
        "CoMeRe": 0,
        "CFPP": 0,
        "CRFP": 0,
        "CLAPI": 0,
    }

    # Recherche récursive
    files = sorted(
        p
        for p in root.rglob("*.orfeo")
        if p.is_file()
        and wanted_file(p)
    )

    print()
    print(
        f"{len(files):,} fichiers .orfeo sélectionnés."
    )

    if not files:
        print()
        print(
            "[ATTENTION] Aucun fichier .orfeo trouvé."
        )
        print(
            "Vérifie que --input pointe vers le dossier "
            "racine contenant CoMeRe / CFPP / CRFP / CLAPI."
        )

        return data, counts_by_corpus

    seen = set()

    for i, path in enumerate(files, 1):

        corpus = identify_corpus(path)

        if corpus == "CoMeRe":
            group = "written"
        else:
            group = "oral"

        file_added = 0

        for sentence in read_orfeo(path):

            sentence = normalize(sentence)

            if not valid_sentence(sentence):
                continue

            # Déduplication insensible à la casse
            key = sentence.casefold()

            if key in seen:
                continue

            seen.add(key)

            data[group].append(
                {
                    "text": sentence,
                    "corpus": corpus,
                }
            )

            counts_by_corpus[corpus] += 1
            file_added += 1

        print(
            f"[{i:>4}/{len(files)}] "
            f"{corpus:<6} | "
            f"{path.name} | "
            f"+{file_added:,}"
        )

    return data, counts_by_corpus


# ------------------------------------------------------------------
# Échantillonnage
# ------------------------------------------------------------------

def sample_data(
    data,
    written_n,
    oral_n,
    seed
):

    rng = random.Random(seed)

    written = data["written"].copy()
    oral = data["oral"].copy()

    rng.shuffle(written)
    rng.shuffle(oral)

    written = written[
        :min(
            written_n,
            len(written)
        )
    ]

    oral = oral[
        :min(
            oral_n,
            len(oral)
        )
    ]

    rows = []

    for item in written:

        rows.append({
            "text": item["text"],
            "source": "ORFEO_CoMeRe",
            "register": "written_conversational",
            "label": 0,
            "corpus": item["corpus"],
        })

    for item in oral:

        rows.append({
            "text": item["text"],
            "source": "ORFEO_oral_FR",
            "register": "oral",
            "label": 0,
            "corpus": item["corpus"],
        })

    rng.shuffle(rows)

    return rows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extraction FI depuis CEFC-ORFEO : "
            "CoMeRe + CFPP + CRFP + CLAPI."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help=(
            "Dossier racine contenant les "
            "sous-corpus CEFC-ORFEO."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "../data/orfeo_fi.csv"
        ),
    )

    parser.add_argument(
        "--written_n",
        type=int,
        default=15000,
        help=(
            "Nombre maximal de phrases "
            "CoMeRe à conserver."
        ),
    )

    parser.add_argument(
        "--oral_n",
        type=int,
        default=20000,
        help=(
            "Nombre maximal de phrases "
            "orales à conserver."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # --------------------------------------------------------------
    # Vérification du dossier
    # --------------------------------------------------------------

    if not args.input.exists():

        raise SystemExit(
            f"Dossier introuvable : "
            f"{args.input}"
        )

    if not args.input.is_dir():

        raise SystemExit(
            f"--input doit être un dossier : "
            f"{args.input}"
        )

    # --------------------------------------------------------------
    # Extraction
    # --------------------------------------------------------------

    data, counts = extract(
        args.input
    )

    print()
    print("=" * 60)
    print(
        "PHRASES DISPONIBLES AVANT ÉCHANTILLONNAGE"
    )
    print("=" * 60)

    print(
        f"CoMeRe : {counts['CoMeRe']:,}"
    )

    print(
        f"CFPP   : {counts['CFPP']:,}"
    )

    print(
        f"CRFP   : {counts['CRFP']:,}"
    )

    print(
        f"CLAPI  : {counts['CLAPI']:,}"
    )

    print()

    print(
        f"Écrit total : "
        f"{len(data['written']):,}"
    )

    print(
        f"Oral total  : "
        f"{len(data['oral']):,}"
    )

    # --------------------------------------------------------------
    # Sampling
    # --------------------------------------------------------------

    rows = sample_data(
        data=data,
        written_n=args.written_n,
        oral_n=args.oral_n,
        seed=args.seed,
    )

    # --------------------------------------------------------------
    # Sortie
    # --------------------------------------------------------------

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "text",
        "source",
        "register",
        "label",
        "corpus",
    ]

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------------------
    # Résumé
    # --------------------------------------------------------------

    written_final = sum(
        row["register"]
        == "written_conversational"
        for row in rows
    )

    oral_final = sum(
        row["register"]
        == "oral"
        for row in rows
    )

    print()
    print("=" * 60)
    print("RÉSULTAT FINAL")
    print("=" * 60)

    print(
        f"Written : {written_final:,}"
    )

    print(
        f"Oral    : {oral_final:,}"
    )

    print(
        f"Total   : {len(rows):,}"
    )

    print()
    print(
        f"Sortie : {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()