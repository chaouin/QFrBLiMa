# download_vgq.py

import re
import time
from pathlib import Path
import os
from urllib.parse import urljoin

import pymupdf
import pandas as pd
import requests
import spacy
from bs4 import BeautifulSoup

BASE = "https://www.vgq.qc.ca"

CATEGORY_URLS = [
    # Audit financier / audit de performance
    "https://www.vgq.qc.ca/fr/publications?annee=&typeRapport=RA",

    # Rapports annuels de gestion
    "https://www.vgq.qc.ca/fr/publications?annee=&typeRapport=RG",

    # Commissaire au développement durable
    "https://www.vgq.qc.ca/fr/publications?annee=&lang=fr&typeRapport=CDD",
]

OUT_DIR = Path(os.environ.get("QFR_CORPUS_ROOT", str(Path(__file__).resolve().parents[3] / "data" / "training_corpus" / "raw"))).expanduser().resolve() / "data"
PDF_DIR = OUT_DIR / "vgq"
PDF_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 research-corpus-builder/1.0"
}

nlp = spacy.load(
    "fr_core_news_sm",
    disable=["ner", "tagger", "lemmatizer"]
)

nlp.max_length = 3_000_000


def get_soup(url):
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def get_publication_pages():
    pages = set()

    for url in CATEGORY_URLS:
        soup = get_soup(url)

        for a in soup.find_all("a", href=True):
            href = a["href"]

            if re.fullmatch(r"/fr/publications/\d+", href):
                pages.add(urljoin(BASE, href))

    return sorted(pages)


def find_full_report_pdf(publication_url):
    soup = get_soup(publication_url)

    # Première préférence : lien explicitement nommé "Rapport complet"
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True).lower()

        if "rapport complet" in text:
            href = a["href"]

            if ".pdf" in href.lower():
                return urljoin(BASE, href)

    # secours : premier PDF contenant "rapport"
    for a in soup.find_all("a", href=True):
        href = a["href"]

        if ".pdf" in href.lower() and "rapport" in href.lower():
            return urljoin(BASE, href)

    return None


def download_pdf(url, destination):
    if destination.exists():
        return

    r = requests.get(url, headers=HEADERS, timeout=120)
    r.raise_for_status()
    destination.write_bytes(r.content)


def pdf_to_text(path):
    doc = pymupdf.open(path)

    pages = []

    for page in doc:
        text = page.get_text("text")
        pages.append(text)

    return "\n".join(pages)


def clean_text(text):
    text = text.replace("\u00ad", "")
    text = text.replace("\xa0", " ")

    # mots coupés en fin de ligne
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # retours de ligne internes
    text = re.sub(r"(?<![.!?])\n(?!\n)", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def sentence_split(text):
    doc = nlp(text)

    sentences = []

    for sent in doc.sents:
        s = sent.text.strip()

        # filtres minimaux
        if len(s) < 30:
            continue

        words = s.split()

        if len(words) < 5:
            continue

        if len(words) > 120:
            continue

        # enlever lignes essentiellement numériques
        alpha_ratio = sum(c.isalpha() for c in s) / max(len(s), 1)

        if alpha_ratio < 0.55:
            continue

        sentences.append(s)

    return sentences


def main():
    publication_pages = get_publication_pages()[:40]

    print(f"{len(publication_pages)} publications trouvées.")

    rows = []

    for i, publication_url in enumerate(publication_pages, start=1):
        publication_id = publication_url.rstrip("/").split("/")[-1]

        print(f"[{i}/{len(publication_pages)}] {publication_url}")

        pdf_url = find_full_report_pdf(publication_url)

        if not pdf_url:
            print("  Aucun rapport complet trouvé.")
            continue

        pdf_path = PDF_DIR / f"vgq_{publication_id}.pdf"

        try:
            download_pdf(pdf_url, pdf_path)

            raw_text = pdf_to_text(pdf_path)
            text = clean_text(raw_text)

            sentences = sentence_split(text)

            for sentence in sentences:
                rows.append({
                    "text": sentence,
                    "variety": "FQ",
                    "genre": "government_report",
                    "source": "VGQ",
                    "document_id": publication_id,
                    "document_url": publication_url,
                    "pdf_url": pdf_url,
                })

            print(f"  {len(sentences)} phrases")

        except Exception as e:
            print(f"  ERREUR: {e}")

        time.sleep(0.5)

    df = pd.DataFrame(rows)

    df = df.drop_duplicates(subset=["text"])

    out = PDF_DIR / "vgq_sentences.csv"
    df.to_csv(out, index=False)

    print()
    print(f"Total : {len(df):,} phrases")
    print(f"Sauvegardé : {out}")


if __name__ == "__main__":
    main()