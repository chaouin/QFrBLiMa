# Regional classifier training corpus

This directory documents the supervised corpus used for the five encoder classifiers. The raw texts and generated splits are not included. The reported statistics are those of the original training corpus, not a newly reconstructed copy. The scientific notation is **MFr (Metropolitan French)** and **QFr (Quebec French)**. For backward compatibility, legacy identifiers `FI` (0) and `FQ` (1) are retained in the dataset, code, prompts and historical results. These denote the same operational categories, not a claim that International French and Metropolitan French are linguistically synonymous.

## Sources

| Genre | MFr | QFr |
|---|---|---|
| Encyclopedic | French Wikipedia and Wikisource | Quebec portal articles on Wikipedia |
| Parliamentary | French National Assembly | National Assembly of Quebec |
| Administrative | Vie-publique | Auditor General of Quebec (VGQ) |
| Oral | ORFEO | CRIFUQ |
| Informal Web | CoMeRe | Quebec Reddit and Depotoir.ca |
| Journalistic | MLSUM | CN2i and Radio-Canada |

FreCDo informed part of the MFr source selection; other resources were selected to match the QFr genres.

### Provenance of the shared resources

Five QFr resources are documented in [Khan et al. (2026)](https://aclanthology.org/2026.lrec-1.840/): Quebec Wikipedia, CN2i–Le Soleil, CRIFUQ, Quebec Reddit and Depotoir.ca. The shared research collection also includes Radio-Canada, Quebec parliamentary transcripts and French Wikipedia/Wikisource, which are not described in that publication. The exact correspondence between the downloaded versions and the versions used in the published study has not been established.

The prepared TXT files were downloaded manually from the shared collection hosted in S3 storage on Université Laval's VALERIA infrastructure, with research access granted by Richard Khoury. VALERIA is the hosting infrastructure, not the corpus producer. The other resources were obtained independently; the included VGQ collection script provides a public-source acquisition route, but its use to produce the historical training input has not been independently established.

The contribution of this repository is the selection, harmonization, cleaning, balancing and partitioning of existing resources for supervised regional classification. See [sources.yaml](sources.yaml) for source-specific provenance, licensing and upstream preprocessing.

### Depotoir input verification

The historical input `data/forums/forums_qc_sentences.txt` is a renamed, byte-identical copy of `commentaires_depotoir.txt` from the shared collection. Both files have the following SHA-256 digest:

```text
1c4dcce0ef5bb7be6ee2e23e426425df155b69706cc1e8c9573d2f21b2f325c8
```

The comparison also found 1,035,062 identical text units (100% coverage after splitting on line breaks and `</s>` separators). No additional comment corpus is needed to reproduce this particular input.

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

| Partition | MFr | QFr | Total |
|---|---:|---:|---:|
| Train | 375644 | 375644 | 751288 |
| Validation | 46884 | 46884 | 93768 |
| Test | 46714 | 46714 | 93428 |
| Total | 469242 | 469242 | 938484 |

See [corpus_statistics.csv](corpus_statistics.csv) for the breakdown by genre and variety. Each genre is balanced between MFr and QFr within each partition; the genres themselves are not necessarily equal in size. Only the 751,288 training examples were used for gradient updates.

## Data availability and licensing

Following [Khan et al. (2026)](https://aclanthology.org/2026.lrec-1.840/), we do not publicly redistribute the supervised corpus, including raw source texts and generated training, validation and test splits, due to copyright and source-specific usage restrictions. The repository provides construction scripts, source documentation and aggregate statistics. Researchers must obtain the original resources independently and comply with their applicable access and usage conditions.

The shared corpus metadata includes Creative Commons licenses and non-commercial research restrictions. Independently acquired resources are subject to their respective licenses and terms, including Etalab Open Licence 2.0, CC BY 4.0 (CoMeRe), CC BY-NC-SA 4.0 (CEFC-ORFEO), non-commercial research restrictions (MLSUM), and source-specific terms (VGQ). See [sources.yaml](sources.yaml) for details. The scope of the Vie-publique license for report text, the Radio-Canada usage conditions, and any applicable subcorpus-specific restrictions remain to be confirmed. The repository license does not supersede the rights associated with the source data. We intend to release the five fine-tuned classifiers on Hugging Face, subject to confirmation of the applicable pretrained-model licenses and training-data usage permissions. Access to the shared collection does not by itself establish permission to redistribute source texts or model weights.

## Limitations

- Regional labels are source-level proxies, not sentence-level linguistic annotations. French Wikipedia/Wikisource is pan-Francophone, while Quebec Wikipedia articles are selected by topic association with Quebec rather than author-level dialect annotation.
- Genre balancing reduces genre-related confounding but does not eliminate source-specific topics, collection methods or preprocessing artifacts. The upstream Reddit collection also used Quebec French lexical markers during selection.
- Document-level separation is applied when identifiers are available; otherwise row-level grouping is used. Exact text overlap across partitions is checked by the builder, but this does not establish complete independence at the original document level.
- The precise historical input snapshots, preparation commands and cache state have not all been recovered. Exact reconstruction of the original corpus has not been verified. The ANDQ metadata also contains an unresolved inconsistency concerning its parent resource.

## References

Khan, E., Saidani, F., Van Esbroeck, O., Khoury, R., & Kosseim, L. (2026). *Low-Resource Dialect Adaptation of Large Language Models: A French Dialect Case-Study*. LREC 2026. https://aclanthology.org/2026.lrec-1.840/

Additional resource links and licensing references are listed in [sources.yaml](sources.yaml).
