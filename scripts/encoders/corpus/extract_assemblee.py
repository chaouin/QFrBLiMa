#!/usr/bin/env python3

import argparse
import csv
import html
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path


NS = {
    "an": "http://schemas.assemblee-nationale.fr/referentiel"
}


def normalize(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\u202f", " ")

    # Supprime les indications éditoriales qui peuvent rester
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def element_text(element):
    """
    Récupère tout le texte d'un élément, y compris celui contenu
    dans <italique>, <exposant>, <br/>, etc.
    """
    if element is None:
        return ""

    return normalize(" ".join(element.itertext()))


def remove_stage_directions(text: str) -> str:
    """
    Enlève les indications du compte rendu :
    (Applaudissements...)
    (Exclamations...)
    (Mêmes mouvements.)
    etc.
    """
    patterns = [
        r"\([^()]*applaudissement[^()]*\)",
        r"\([^()]*exclamation[^()]*\)",
        r"\([^()]*protestation[^()]*\)",
        r"\([^()]*rires?[^()]*\)",
        r"\([^()]*murmures?[^()]*\)",
        r"\([^()]*mêmes mouvements[^()]*\)",
        r"\([^()]*sourires?[^()]*\)",
    ]

    for pattern in patterns:
        text = re.sub(
            pattern,
            " ",
            text,
            flags=re.IGNORECASE
        )

    return normalize(text)


def split_sentences(text: str):
    """
    Segmentation légère.
    Suffisante pour créer le corpus brut.

    Idéalement, on appliquera ensuite exactement le même
    segmenter à FQ et FI.
    """
    text = normalize(text)

    if not text:
        return []

    return re.split(
        r"(?<=[.!?…])\s+(?=[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŒ«])",
        text
    )


def valid_sentence(sentence: str) -> bool:
    sentence = normalize(sentence)
    words = sentence.split()

    # Évite les fragments trop courts
    if len(words) < 5:
        return False

    # Évite les très longs blocs dus à une mauvaise segmentation
    if len(words) > 100:
        return False

    if len(sentence) < 25:
        return False

    if not re.search(r"[A-Za-zÀ-ÿ]", sentence):
        return False

    return True


def extract_date(root):
    elem = root.find(
        ".//an:metadonnees/an:dateSeanceJour",
        NS
    )

    return normalize(elem.text) if elem is not None else ""


def extract_session(root):
    elem = root.find(
        ".//an:metadonnees/an:session",
        NS
    )

    return normalize(elem.text) if elem is not None else ""


def extract_from_xml(xml_path: Path):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as e:
        print(f"[WARN] XML invalide : {xml_path}: {e}")
        return []

    date = extract_date(root)
    session = extract_session(root)

    rows = []

    # On ne regarde que les paragraphes dans <contenu>.
    paragraphs = root.findall(
        ".//an:contenu//an:paragraphe",
        NS
    )

    for paragraph in paragraphs:

        code_grammaire = paragraph.attrib.get(
            "code_grammaire", ""
        )

        code_parole = paragraph.attrib.get(
            "code_parole", ""
        )

        # -----------------------------------------------------
        # Filtre principal :
        #
        # PAROLE_1_2 = vraie intervention de l'orateur
        #
        # PAROLE_1_1 correspond souvent à :
        # "La parole est à Mme X..."
        #
        # INTERRUPTIONS exclues.
        # -----------------------------------------------------

        if code_grammaire != "PAROLE_GENERIQUE":
            continue

        if code_parole != "PAROLE_1_2":
            continue

        speaker_el = paragraph.find(
            "./an:orateurs/an:orateur/an:nom",
            NS
        )

        quality_el = paragraph.find(
            "./an:orateurs/an:orateur/an:qualite",
            NS
        )

        text_el = paragraph.find(
            "./an:texte",
            NS
        )

        speaker = (
            normalize(speaker_el.text)
            if speaker_el is not None
            else ""
        )

        quality = (
            normalize(quality_el.text)
            if quality_el is not None
            else ""
        )

        text = element_text(text_el)
        text = remove_stage_directions(text)

        if not text:
            continue

        for sentence in split_sentences(text):

            sentence = normalize(sentence)

            if not valid_sentence(sentence):
                continue

            rows.append({
                "text": sentence,
                "speaker": speaker,
                "quality": quality,
                "date": date,
                "session": session,
                "file": xml_path.name,
                "source": "assemblee_nationale_fr",
                "register": "formal_institutional",
                "label": 0,
            })

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Dossier contenant les XML de l'Assemblée nationale",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "../data/assemblee_nationale_fr.csv"
        ),
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=35000,
        help="Nombre maximal de phrases à conserver",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    xml_files = sorted(
        args.input.rglob("*.xml")
    )

    print(
        f"{len(xml_files):,} fichiers XML trouvés."
    )

    rows = []
    seen = set()

    for i, xml_path in enumerate(xml_files, 1):

        extracted = extract_from_xml(xml_path)

        for row in extracted:

            key = row["text"].lower()

            if key in seen:
                continue

            seen.add(key)
            rows.append(row)

        if i % 100 == 0:
            print(
                f"{i:,}/{len(xml_files):,} fichiers | "
                f"{len(rows):,} phrases"
            )

    print()
    print(
        f"Phrases uniques avant sampling : "
        f"{len(rows):,}"
    )

    # Reproductible
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    if args.sample > 0:
        rows = rows[:args.sample]

    print(
        f"Phrases finales : {len(rows):,}"
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "text",
        "speaker",
        "quality",
        "date",
        "session",
        "file",
        "source",
        "register",
        "label",
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

    print(f"Écrit dans : {args.output}")


if __name__ == "__main__":
    main()