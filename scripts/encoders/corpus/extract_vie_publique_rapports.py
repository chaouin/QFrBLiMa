#!/usr/bin/env python3

import argparse
import csv
import html
import json
import random
import re
from pathlib import Path


def normalize(text: str) -> str:
    if not text:
        return ""

    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\u202f", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text: str):
    text = normalize(text)

    if not text:
        return []

    return re.split(
        r"(?<=[.!?…])\s+(?=[A-ZÀÂÄÇÉÈÊËÎÏÔÖÙÛÜŒ«])",
        text,
    )


def valid_sentence(text: str) -> bool:
    words = text.split()

    if len(words) < 5:
        return False

    if len(words) > 80:
        return False

    if len(text) < 25:
        return False

    if not re.search(r"[A-Za-zÀ-ÿ]", text):
        return False

    alpha = len(re.findall(r"[A-Za-zÀ-ÿ]", text))

    if alpha / max(len(text), 1) < 0.50:
        return False

    return True


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    print(f"Lecture : {args.input}")

    with args.input.open(
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    print(f"Rapports : {len(data):,}")

    rows = []
    seen = set()

    for report in data:

        report_id = str(report.get("id", ""))
        title = normalize(report.get("titre", ""))
        date = (
            report.get("date_diffusion")
            or report.get("date_remise")
            or ""
        )

        # On utilise le vrai contenu rédactionnel disponible,
        # pas le sommaire.
        fields = [
            ("presentation", report.get("presentation")),
            ("synthese", report.get("synthese")),
            ("extrait", report.get("extrait")),
        ]

        for field_name, content in fields:

            if not content:
                continue

            for sentence in split_sentences(content):

                sentence = normalize(sentence)

                if not valid_sentence(sentence):
                    continue

                key = sentence.casefold()

                if key in seen:
                    continue

                seen.add(key)

                rows.append({
                    "text": sentence,
                    "document_id": report_id,
                    "document_title": title,
                    "date": date,
                    "field": field_name,
                    "source": "vie_publique_rapports",
                    "register": "formal_expository",
                    "label": 0,
                })

    print(
        f"Phrases uniques disponibles : "
        f"{len(rows):,}"
    )

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    if len(rows) < args.sample:
        print(
            f"[ATTENTION] Seulement {len(rows):,} "
            f"phrases disponibles pour une cible de "
            f"{args.sample:,}."
        )

    rows = rows[:args.sample]

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields = [
        "text",
        "document_id",
        "document_title",
        "date",
        "field",
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

    print(f"Phrases finales : {len(rows):,}")
    print(f"Sortie : {args.output}")


if __name__ == "__main__":
    main()