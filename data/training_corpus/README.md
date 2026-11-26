# Regional classifier training corpus

This directory documents the supervised corpus used for the five encoder classifiers. The raw texts and generated splits are not included. The reported statistics are those of the original training corpus, not a newly reconstructed copy. FI (0) and FQ (1) are the code labels corresponding to MFr and QFr in the paper.

## Sources

| Genre | FI | FQ |
|---|---|---|
| Encyclopedic | Wikipedia FR | Wikipedia Quebec |
| Parliamentary | French National Assembly | National Assembly of Quebec |
| Administrative | Vie-publique | Auditor General of Quebec (VGQ) |
| Oral | ORFEO | CRIFUQ |
| Informal Web | CoMeRe | Quebec Reddit and forums |
| Journalistic | MLSUM | CN2i and Radio-Canada |

FreCDo informed part of the FI source selection; other sources were selected to match the QFr genres. QFr resources were obtained through VALERIA according to the project records. Exact resource versions, access conditions, and redistribution permissions remain to be confirmed; the included VGQ collection script also offers a public-source acquisition route, not evidence that it produced the actual training input.

## Construction from prepared inputs

The main builder is `scripts/encoders/corpus/build_classifier_corpus.py`. It preserves the original seed (42), sampling cap (100,000 per variety and genre), target split ratios (80/10/10), filtering, balancing and splitting logic. Document grouping is used where identifiers are available; some sources fall back to row-level identifiers. The original builder could reuse genre caches; this release rebuilds by default. To reuse compatible historical caches, set `QFR_REUSE_GENRE_CACHE=1` explicitly. The actual cache state used for the original training corpus has not been established.

From the repository root, install the corpus dependencies (pandas, requests, spacy, lxml, pymupdf, beautifulsoup4); the scripts using spaCy additionally require `fr_core_news_sm`. Exact historical dependency versions are not yet established.

```bash
export QFR_CORPUS_ROOT=/path/to/corpus_workspace
python scripts/encoders/corpus/build_classifier_corpus.py
```

The workspace must contain the following prepared inputs under `data/`:

```text
data/
  wiki/balanced/wikipedia_fr_balanced.csv
  wiki/balanced/wikipedia_qc_balanced.csv
  assemblee_fr/assemblee_fr_sentences.csv
  assemblee_qc/assemblee_qc_sentences.txt
  vie_publique/vie_publique_sentences.json
  vgq/vgq_sentences.csv
  orfeo/orfeo_oral_sentences.csv
  crifuq/crifuq_sentences.txt
  comere/comere_informal_web_sentences.csv
  reddit/reddit_qc_sentences.txt
  forums/forums_qc_sentences.txt
  mlsum/processed/mlsum_train_sentences.csv
  cn2i/cn2i_sentences.txt
  radio_canada/radio_canada_sentences.txt
```

CSV inputs need a `text` column; recognized document identifiers are preserved where available. The builder reads Vie-publique as a JSON array of report objects with `id` and text fields (`titre`, `presentation`, `sommaire`, `synthese`, `extrait`). See the builder for precise format rules. Outputs are placed in `$QFR_CORPUS_ROOT/data/fq_fi_classifier_corpus/`.

## Source preparation scripts

The accompanying collection/extraction scripts are retained as historical preparation utilities, not a verified end-to-end reconstruction workflow. The Wikipedia utility expects prepared FR parts and a cleaned Quebec text file. The Vie-publique extractor emits CSV, not the JSON expected by the final builder. The ORFEO utility can emit both oral and written rows and its default sample is smaller than the reported final oral corpus; the exact historical command and filtering must be recovered. The VGQ output path has been aligned with the final builder. No historical sampling parameters have been changed. The exact commands, input snapshots, and intermediate file provenance need confirmation before claiming exact reconstruction.

## Reported corpus sizes

| Partition | FI | FQ | Total |
|---|---:|---:|---:|
| Train | 375644 | 375644 | 751288 |
| Validation | 46884 | 46884 | 93768 |
| Test | 46714 | 46714 | 93428 |
| Total | 469242 | 469242 | 938484 |

See `corpus_statistics.csv` for the breakdown by genre and variety. No raw source data, generated training texts, or checkpoints are included in this addition.
