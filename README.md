# QFrBLiMa: a Quebec French Benchmark of Linguistic Markers

**QFrBLiMa** is a contrastive benchmark designed to evaluate language models' ability to recognize documented regional associations of linguistic markers in **Quebec French (QFr)** and **Metropolitan French (MFr)**. We evaluate two widely used classification strategies: zero-shot prompting of large language models (LLMs) and supervised fine-tuning of BERT-family classifiers.

Unlike grammatical minimal-pair benchmarks, QFrBLiMa evaluates **regional linguistic recognition rather than grammatical acceptability**. Both sentence variants may be grammatically correct.

The benchmark comprises **1,205 contrastive sentence pairs**, covering **241 linguistic markers** across 12 thematic categories, with five contexts per linguistic marker. The linguistic markers are grounded in terminology documented by the *Grand dictionnaire terminologique* (GDT), maintained by the Office québécois de la langue française (OQLF).

**Dataset:** [Hugging Face](https://huggingface.co/datasets/chaouin/qfrblima) | **Models:** [Fine-tuned classifiers](#supervised-encoders) | **Paper:** Preprint forthcoming

## Terminology and Notation

The scientific notation is **MFr (Metropolitan French)** and **QFr (Quebec French)**. For backward compatibility, legacy identifiers `FI` (0) and `FQ` (1) are retained in the dataset, code, prompts and historical results. These denote the same operational categories.

A linguistic marker consists of two lexical or terminological variants with the same intended meaning and different documented regional associations. A variant is an individual word or expression within a linguistic marker. A context is the surrounding linguistic material in which a variant occurs. A sentence pair consists of the two sentences obtained by inserting the respective variants into the same context, with grammatical adjustments where necessary.

## Dataset Statistics

| Property | Value |
| --- | --- |
| Language | French (Quebec French / Metropolitan French) |
| Linguistic markers | 241 |
| Sentence pairs | 1,205 |
| Contexts per linguistic marker | 5 |
| Thematic categories | 12 |
| Evaluation | Zero-shot LLM judgment and supervised encoder scoring |
| Candidate orders | Original and inverted (LLMs only) |
| Models evaluated | 8 LLMs and 5 encoder classifiers |
| LLM judgments | 19,280 |
| Encoder pair rankings | 6,025 |
| Source | OQLF Grand dictionnaire terminologique |
| License | CC BY-NC-SA 4.0 |

## Dataset Availability & Usage

QFrBLiMa is available on [Hugging Face](https://huggingface.co/datasets/chaouin/qfrblima) and directly in [`data/qfrblima.csv`](data/qfrblima.csv).

### Loading the Dataset

Install the Hugging Face `datasets` library:

```bash
pip install datasets
```

**From Hugging Face:**

```python
from datasets import load_dataset

dataset = load_dataset("chaouin/qfrblima", split="test")

print(dataset[0])
print(f"Pairs: {len(dataset)}")
print(f"Markers: {len(set(dataset['feature_id']))}")
```

**From this repository:**

```python
from datasets import load_dataset

dataset = load_dataset(
    "csv",
    data_files={"test": "data/qfrblima.csv"},
    split="test",
)

print(dataset[0])
```

All 1,205 pairs belong to a single evaluation split (`test`), without training or validation partitions.

### Dataset Structure

Each pair contains two sentences intended to express the same meaning, with a targeted regional linguistic contrast.

| Metropolitan French (MFr) | Quebec French (QFr) |
| --- | --- |
| J'ai terminé le **puzzle** de mille pièces hier soir. | J'ai terminé le **casse-tête** de mille pièces hier soir. |
| Passe-moi la **clé plate** de douze millimètres. | Passe-moi la **clé ouverte** de douze millimètres. |

The dataset contains the following fields:

| Field | Description |
| --- | --- |
| `pair_id` | Unique sentence-pair identifier |
| `feature_id` | Linguistic marker identifier |
| `domaine` | Terminological domain |
| `text_FI` | MFr sentence |
| `text_FQ` | QFr sentence |
| `label_FI` | MFr label (0) |
| `label_FQ` | QFr label (1) |
| `marker_FI` | Target MFr variant |
| `marker_FQ` | Target QFr variant |
| `source` | OQLF GDT source URL |
| `concept_id` | Concept identifier |

## Benchmark Construction

QFrBLiMa was constructed using January 2026 snapshots of terminology from the OQLF's *Grand dictionnaire terminologique*.

Linguistic markers were automatically extracted from terminological data, followed by manual review and cleaning to remove redundant forms and retain documented regional associations.

Five sentence contexts were manually constructed for each of the **241 retained linguistic markers**, resulting in **1,205 contrastive sentence pairs**.

The linguistic markers are grouped into 12 mutually exclusive thematic categories. The complete mapping from principal terminological domains to thematic categories is available in [`data/domain_to_broad_category.csv`](data/domain_to_broad_category.csv).

The benchmark reflects documented regional linguistic associations rather than exclusive geographic usage. The constructed sentences were not independently validated by external annotators for naturalness, semantic equivalence or sense disambiguation.

The source data and extraction code are available in [`data/`](data/) and [`scripts/extract_oqlf_markers.py`](scripts/extract_oqlf_markers.py).

## Evaluation Method

### LLMs: zero-shot contrastive judgment

Models receive two anonymized sentence candidates (A and B) without access to the gold variety labels or expected linguistic markers.

The task is to identify which candidate exhibits stronger Quebec French linguistic characteristics, or select `EQUIVALENT`.

Five prediction labels are permitted:

- `A_MUCH_MORE_FQ`
- `A_SLIGHTLY_MORE_FQ`
- `EQUIVALENT`
- `B_SLIGHTLY_MORE_FQ`
- `B_MUCH_MORE_FQ`

The structured response additionally contains the dominant linguistic dimension, evidence spans and a concise linguistic rationale.

A prediction is correct when the model identifies the documented QFr candidate. `EQUIVALENT` is not counted as correct for these contrastive items.

All eight LLMs receive one sentence pair per request, with one completed judgment retained per pair and candidate order.

All sentence pairs are evaluated in both candidate orders:

- **Mean single-order accuracy:** average accuracy across the two candidate presentations.
- **Order-robust accuracy:** proportion of sentence pairs correctly identified in both presentations.

The evaluation prompt and output schema are available in [`qfrblima/protocol.py`](qfrblima/protocol.py).

### Encoders: independent sentence scoring

The five encoder classifiers are fine-tuned on a separate regional classification corpus. Each sentence is scored independently using the class-1 softmax score (QFr). These scores have not been calibrated.

**Strict pairwise accuracy** measures the proportion of sentence pairs for which the QFr sentence receives a strictly higher QFr score than its MFr counterpart. Ties count as incorrect. Candidate-order inversion is not required for this protocol.

Sentence-level accuracy and macro-F1 are also reported using a classification threshold of 0.5. The implementation is available in [`scripts/encoders/evaluate_classifiers.py`](scripts/encoders/evaluate_classifiers.py).

### Supervised training corpus

The separate corpus contains 938,484 examples across six genres, balanced between MFr and QFr within each genre and partition. Genre sizes differ. QFrBLiMa is used only for external evaluation, not for classifier fine-tuning or checkpoint selection.

| Partition | Examples |
| --- | ---: |
| Training | 751,288 |
| Validation | 93,768 |
| Test | 93,428 |
| Total | 938,484 |

The classifiers were trained for three epochs with seed 42, AdamW, a learning rate of 2e-5, linear scheduling with a warmup ratio of 0.06, an effective batch size of 32, a maximum sequence length of 128 and weight decay of 0.01.

Checkpoints were selected using validation macro-F1 (epoch 2 for ModernCamemBERT and epoch 3 for the other four models).

See [`data/training_corpus/README.md`](data/training_corpus/README.md) for source provenance, construction instructions, aggregate statistics and reconstruction limitations.

## Experiments & Results

### LLMs

We evaluated **eight LLMs** (three API-based and five open-weight models) using both original and inverted candidate orders.

| Model | Mean single-order accuracy | Order-robust accuracy |
| --- | ---: | ---: |
| Gemini 3.8 Flash | 94.69% | 93.53% |
| Claude Opus 4.8 | 84.48% | 76.18% |
| GPT-5.4 | 75.89% | 66.56% |
| Gemma 4 12B IT | 60.62% | 48.38% |
| Mistral NeMo 12B Instruct | 53.61% | 11.78% |
| Qwen3 8B | 47.63% | 6.56% |
| Aya Expanse 8B | 46.60% | 6.97% |
| Llama 3.1 8B Instruct | 32.74% | 10.29% |

Mean single-order accuracy averages accuracy across the two candidate presentations. Predictions and analyses are available in [`results/predictions/`](results/predictions/) and [`results/analysis/`](results/analysis/).

### Supervised encoders

The five fine-tuned classifiers are available on Hugging Face. Each model card documents training, evaluation and intended use.

| Model | Internal test macro-F1 | QFrBLiMa sentence accuracy | QFrBLiMa sentence macro-F1 | Strict pairwise accuracy (95% CI) |
| --- | ---: | ---: | ---: | ---: |
| [CamemBERT](https://huggingface.co/chaouin/qfrblima-camembert) | 97.56% | 59.38% | 59.20% | 72.45% (67.97–76.68%) |
| [CamemBERTv2](https://huggingface.co/chaouin/qfrblima-camembertv2) | 97.07% | 57.39% | 54.89% | 70.37% (65.73–74.85%) |
| [ModernCamemBERT](https://huggingface.co/chaouin/qfrblima-moderncamembert) | 97.09% | 56.97% | 53.24% | 69.38% (64.98–73.61%) |
| [BERT-base-cased](https://huggingface.co/chaouin/qfrblima-bert-base-cased) | 95.45% | 53.78% | 52.44% | 58.01% (52.78–63.07%) |
| [RoBERTa-base](https://huggingface.co/chaouin/qfrblima-roberta-base) | 96.36% | 54.27% | 53.68% | 59.67% (54.94–64.32%) |

Internal training results, external predictions and statistical analyses are available in [`results/encoders/`](results/encoders/).

### Preference Strength and Threshold Sensitivity

The preference-strength analysis distinguishes five outcomes:

- Strong correct preference
- Slight correct preference
- No preference
- Slight incorrect preference
- Strong incorrect preference

For LLMs, these categories follow the returned preference labels and candidate correctness, using both candidate orders (2,410 judgments per model).

For encoders, preference strength is derived from the score margin Δ, defined as the QFr score assigned to the QFr sentence minus the QFr score assigned to the MFr sentence.

The main analysis uses the following symmetric thresholds:

| Preference category | Encoder score margin |
| --- | --- |
| Strong correct | Δ ≥ 0.20 |
| Slight correct | 0.05 ≤ Δ < 0.20 |
| No preference | abs(Δ) < 0.05 |
| Slight incorrect | −0.20 < Δ ≤ −0.05 |
| Strong incorrect | Δ ≤ −0.20 |

Sensitivity analysis evaluates 24 valid combinations of no-preference thresholds (0.01, 0.025, 0.05, 0.075, 0.10) and strong-preference thresholds (0.10, 0.15, 0.20, 0.25, 0.30), requiring the no-preference threshold to be strictly smaller than the strong-preference threshold.

The complete results are available in:

- [`preference_strength.csv`](results/analysis/preference_strength.csv): main preference-strength distributions.
- [`preference_threshold_sensitivity.csv`](results/analysis/preference_threshold_sensitivity.csv): results across all 24 threshold combinations for each encoder.
- [`encoder_margin_distribution.csv`](results/analysis/encoder_margin_distribution.csv): descriptive statistics of absolute encoder score margins.

Explicit LLM equivalence and limited encoder score separation are distinct outcomes. These categories describe preference strength rather than calibrated confidence. Threshold selection does not affect the primary recognition accuracy metrics.

### Contextual and Thematic Recognition

Contextual consistency measures recognition across the five contexts associated with each linguistic marker:

- **Full recognition:** five successful contexts.
- **Partial recognition:** one to four successful contexts.
- **No recognition:** zero successful contexts.

Success is defined by order-robust correctness for LLMs and strict pairwise ranking for encoders.

Thematic recognition is evaluated across the 12 thematic categories. Individual model accuracies weight constituent sentence pairs equally, while group-level means average the evaluated models within each strategy.

The complete thematic results are available in [`thematic_model_results.csv`](results/analysis/thematic_model_results.csv) and [`thematic_means.csv`](results/analysis/thematic_means.csv).

Complementary coverage measures the number of sentence pairs successfully recognized by at least one model within each strategy. This analysis does not represent deployable ensemble performance.

### Evidence Span Identification and Attribution

The evidence-span analysis investigates whether LLMs can identify the relevant lexical or terminological variants independently of their final regional attribution.

An evidence span is a text excerpt returned by the model to support its decision.

Four metrics are reported:

| Metric | Definition |
| --- | --- |
| Grounding | Both evidence spans occur in distinct candidate sentences. |
| Variant identification | Both documented variants are identified, regardless of their assigned regional labels. |
| Span attribution | Both documented variants are identified and correctly attributed to their respective regional varieties. |
| Conditional decision accuracy | The final candidate selection is correct among judgments with successful variant identification. |

Grounding, variant identification and span attribution use all 2,410 judgments per model as their denominator. Conditional decision accuracy uses only judgments with successful variant identification.

Matching uses Unicode NFKC normalization, case folding and punctuation-insensitive tokenization. Missing evidence spans, including equivalence responses, count as unsuccessful identification.

The complete results are available in:

- [`evidence_span_metrics.csv`](results/analysis/evidence_spans/evidence_span_metrics.csv): aggregate metrics and counts for each LLM.
- [`evidence_span_judgments.csv`](results/analysis/evidence_spans/evidence_span_judgments.csv): individual outcomes for all 19,280 judgments.

Automatic evidence-span matching does not establish the semantic validity or faithfulness of model rationales.

### Statistical analysis

Confidence intervals use 20,000 bootstrap replicates clustered by linguistic marker, retaining all five contexts and both candidate orders where applicable.

Within-strategy comparisons use paired marker-level sign-flip permutation tests (20,000 permutations), with Holm correction applied separately across 28 LLM order-robust comparisons and ten encoder strict pairwise comparisons.

Exact paired McNemar tests compare original and inverted marginal accuracy, with Holm correction across the eight LLMs. These supplementary tests do not account for within-marker dependence.

The analysis uses seed 42.

The LLM and encoder protocols differ in their inputs and decision rules. Cross-strategy comparisons are descriptive rather than tests of equivalent performance measures.

Under independent uniform binary guesses without ties or equivalence responses, chance accuracy is 50% for a single presentation or pairwise ordering and 25% for correctness in both LLM presentations. Independent uniform five-label guessing yields 40% single-order accuracy and 16% order-robust accuracy.

For detailed findings and discussion, please refer to the accompanying paper (preprint forthcoming).

## Reproducing the Results

Clone the repository and install the dependencies:

```bash
git clone https://github.com/chaouin/QFrBLiMa.git
cd QFrBLiMa

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The released predictions and evaluation metrics allow the analyses and manuscript figures to be reproduced without API keys, model checkpoints, a GPU or access to the supervised training corpus.

### LLM analysis

To regenerate the randomized candidate presentations:

```bash
python scripts/prepare.py
```

The released candidate-order manifests can be used directly when reproducing the historical analyses.

Reproduce the analyses using the released model predictions:

```bash
python scripts/analyze.py \
  --benchmark data/qfrblima.csv \
  --predictions results/predictions \
  --out results/analysis \
  --bootstrap 20000 \
  --permutations 20000 \
  --seed 42
```

This script validates the original and inverted predictions and computes recognition accuracy, paired outcomes, cluster-bootstrap confidence intervals, statistical comparisons, contextual consistency, thematic results and shared errors.

For new model evaluations, see [`scripts/evaluate.py`](scripts/evaluate.py), [`configs/models.yaml`](configs/models.yaml), and [`slurm/`](slurm/).

### Encoder analysis and cross-family comparison

The released results can be analyzed without checkpoints, a GPU or API access. In the active virtual environment, install the analysis dependencies if needed:

```bash
python -m pip install numpy pandas scipy scikit-learn matplotlib PyYAML
```

From the repository root:

```bash
python scripts/encoders/analyze_classifiers.py \
  --benchmark data/qfrblima.csv \
  --evaluation results/encoders/evaluation \
  --training results/encoders/training \
  --out results/encoders/analysis \
  --bootstrap 20000 \
  --permutations 20000 \
  --seed 42

python scripts/encoders/compare_models.py \
  --benchmark data/qfrblima.csv \
  --predictions results/predictions \
  --evaluation results/encoders/evaluation \
  --out results/encoders/comparison \
  --bootstrap 20000 \
  --seed 42
```

### Preference-strength analysis

Reproduce the preference-strength distributions, threshold sensitivity analysis and encoder margin statistics:

```bash
python scripts/analyze_preference_strength.py
```

The generated CSV files are saved in [`results/analysis/`](results/analysis/).

### Evidence-span analysis

Reproduce the evidence-span metrics and individual judgment outcomes:

```bash
python scripts/analyze_evidence_spans.py \
  --benchmark data/qfrblima.csv \
  --predictions results/predictions \
  --out results/analysis/evidence_spans
```

The generated CSV files reproduce the evidence-span results reported in the accompanying paper.

### Manuscript figures

After generating the main analyses and preference-strength results, reproduce all five manuscript figures:

```bash
python scripts/plot_figures.py
```

This script generates PDF and PNG versions of the following figures:

| Figure | Filename |
| --- | --- |
| Figure 1 — Overall recognition accuracy | `figure1_overall_accuracy` |
| Figure 2 — Preference strength | `figure2_preference_strength` |
| Figure 3 — Candidate-order sensitivity | `figure3_candidate_order` |
| Figure 4 — Contextual consistency | `figure4_contextual_consistency` |
| Figure 5 — Thematic recognition | `figure5_thematic_recognition` |

The figures are saved in [`results/analysis/`](results/analysis/), together with `classification_transfer.csv`, `thematic_means.csv` and `thematic_model_results.csv`.

The analysis scripts and manuscript plotting script are separated so that statistical analyses can be reproduced independently of figure generation.

### Complete manuscript analysis pipeline

To regenerate the main manuscript analysis directory from the released experimental results, run:

```bash
python scripts/analyze.py \
  --benchmark data/qfrblima.csv \
  --predictions results/predictions \
  --out results/analysis \
  --bootstrap 20000 \
  --permutations 20000 \
  --seed 42

python scripts/analyze_preference_strength.py

python scripts/analyze_evidence_spans.py

python scripts/plot_figures.py
```

These commands do not perform model inference or fine-tuning. Encoder-specific statistical analyses and cross-strategy comparison outputs can be regenerated separately using the commands provided above.

### Using the released classifiers

The fine-tuned checkpoints can be used without access to the supervised corpus or retraining. Install PyTorch and Transformers in a compatible environment (see `requirements-encoders.txt` for reference versions).

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "chaouin/qfrblima-camembert"
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

texts = [
    "J'ai terminé le puzzle hier soir.",
    "J'ai terminé le casse-tête hier soir.",
]
inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors="pt",
)

with torch.inference_mode():
    scores = model(**inputs).logits.softmax(dim=-1)[:, 1]

print(scores.tolist())  # QFr scores
print("QFr sentence ranked higher:", (scores[1] > scores[0]).item())
```

Class 0 (`FI`) corresponds to MFr and class 1 (`FQ`) to QFr. Replace `model_id` with any of the five model identifiers linked above.

To evaluate a released checkpoint on the full benchmark, first download it locally (the evaluation script loads local files only):

```bash
hf download chaouin/qfrblima-camembert \
  --local-dir checkpoints/released-camembert

python scripts/encoders/evaluate_classifiers.py \
  --pairs-csv data/qfrblima.csv \
  --model checkpoints/released-camembert \
  --output-dir results/encoders/new_evaluation/camembert \
  --device cpu
```

### Encoder fine-tuning

The reference dependency versions are recorded in `requirements-encoders.txt` and `requirements-corpus.txt`. Use separate Python 3.11 environments for corpus preparation and model training. The dependency lists reflect the inspected environments; installation on other platforms and exact historical versions have not been fully verified. On HPC systems, use the site-compatible PyTorch/CUDA builds.

First obtain and prepare the supervised corpus according to its [documentation](data/training_corpus/README.md). The following example assumes access to the prepared corpus and a locally available pretrained CamemBERT model:

```bash
export QFR_CORPUS_ROOT=/path/to/corpus_workspace

python scripts/encoders/train_classifier.py \
  --data-dir "$QFR_CORPUS_ROOT/data/fq_fi_classifier_corpus" \
  --model /path/to/pretrained/camembert-base \
  --output-dir checkpoints/camembert

python scripts/encoders/evaluate_classifiers.py \
  --pairs-csv data/qfrblima.csv \
  --model checkpoints/camembert/best_model \
  --output-dir results/encoders/new_evaluation/camembert \
  --device cuda
```

The Python scripts use the historical hyperparameter defaults. For cluster execution across all five models, see [`configs/encoders.yaml`](configs/encoders.yaml) and [`slurm/encoders/`](slurm/encoders/). The Slurm scripts require local pretrained models, prepared corpus files, a configured Python environment and site-specific account and resource settings. Exact corpus reconstruction and end-to-end retraining have not been verified.

### Tests

```bash
python -m unittest discover -s tests -p "test_encoders.py" -v
```

These regression tests cover encoder results and statistical procedures without model inference.

## Repository Structure

```text
QFrBLiMa/
├── configs/                         # Model configurations
├── data/
│   ├── qfrblima.csv                 # Benchmark dataset
│   ├── domain_to_broad_category.csv # Thematic category mapping
│   ├── raw/                         # OQLF source data
│   ├── intermediate/                # Extracted terminology
│   ├── manifests/                   # Candidate-order manifests
│   └── training_corpus/             # Supervised corpus documentation
├── qfrblima/                        # Evaluation protocol
├── scripts/
│   ├── extract_oqlf_markers.py
│   ├── prepare.py
│   ├── evaluate.py
│   ├── analyze.py                   # LLM statistical analyses
│   ├── analyze_preference_strength.py
│   ├── analyze_evidence_spans.py
│   ├── plot_figures.py              # Five manuscript figures
│   └── encoders/
│       ├── corpus/                  # Corpus preparation utilities
│       ├── train_classifier.py
│       ├── evaluate_classifiers.py
│       ├── analyze_classifiers.py
│       └── compare_models.py
├── results/
│   ├── predictions/                 # LLM predictions
│   ├── analysis/                    # Manuscript analyses and figures
│   │   └── evidence_spans/           # Evidence-span analysis
│   └── encoders/
│       ├── training/                # Historical fine-tuning results
│       ├── evaluation/              # Encoder predictions and metrics
│       ├── analysis/                # Encoder statistical analyses
│       └── comparison/              # Cross-strategy comparison
├── slurm/                           # Cluster execution
│   └── encoders/
├── tests/                           # Regression and protocol tests
├── requirements.txt
├── requirements-api.txt
├── requirements-local.txt
├── requirements-corpus.txt
├── requirements-encoders.txt
├── LICENSE.txt
└── README.md
```

## Limitations

QFrBLiMa evaluates recognition of documented regional linguistic associations rather than exclusive distinctions between French varieties.

MFr is an operational comparison category, not a homogeneous linguistic variety; some supervised sources are not exclusively metropolitan. The GDT reflects institutional terminology rather than the full distribution of natural Quebec French, particularly informal speech. Sentence contexts were researcher-authored and were not independently validated by external annotators.

Results should therefore be interpreted as contrastive regional linguistic recognition rather than comprehensive Quebec French proficiency.

The supervised corpus uses source-based variety labels, which are imperfect proxies for regional linguistic usage. High internal classification scores do not necessarily imply strong contrastive recognition. The different evaluation protocols also limit direct comparison between LLMs and encoders.

Preference strength and encoder score margins are not calibrated confidence measures. Threshold selection affects preference category distributions but does not affect strict pairwise accuracy. Automatic evidence-span matching does not establish the semantic validity or faithfulness of model rationales.

The thematic categories contain unequal numbers of linguistic markers. Thematic comparisons are descriptive and should not be interpreted as statistically established domain-level differences.

The benchmark and classifiers are intended for research, not for inferring a person's origin or identity. Exact reconstruction of the supervised corpus and historical source versions has not been fully verified. See the data and model availability statement below.

## Data and Model Availability

Following the approach of [Khan et al. (2026)](https://aclanthology.org/2026.lrec-1.840/), we do not publicly redistribute the supervised training corpus, including its raw texts and generated training, validation and test splits, due to copyright and source-specific usage restrictions.

We provide corpus construction scripts, source documentation, aggregate corpus statistics, training configurations, evaluation pipelines, released predictions, statistical analyses and manuscript figures. Source-specific access and usage conditions are documented in [`data/training_corpus/sources.yaml`](data/training_corpus/sources.yaml).

The benchmark is available on [Hugging Face](https://huggingface.co/datasets/chaouin/qfrblima). The five fine-tuned regional classifiers are available through their individual [Hugging Face model repositories](#supervised-encoders). The checkpoints are distributed separately from the supervised corpus and can be used for inference and benchmark evaluation without reconstructing that corpus.

The benchmark and fine-tuned classifiers are also available through the [QFrBLiMa Hugging Face collection](https://huggingface.co/collections/chaouin/qfrblima).

## Citation

If you use QFrBLiMa in your research, please cite the associated paper:

*QFrBLiMa: a Quebec French Benchmark of Linguistic Markers.*

A BibTeX citation and preprint link will be added once the preprint is publicly available.

## License

QFrBLiMa is distributed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

The benchmark incorporates terminology from the *Grand dictionnaire terminologique* (GDT), maintained by the **Office québécois de la langue française (OQLF)** and distributed through **Données Québec**.

Original source attribution and applicable licensing conditions must be preserved. The benchmark license does not grant rights to third-party materials beyond those provided by their respective rights holders. The supervised training sources remain subject to their own licenses and access restrictions. The benchmark license does not automatically extend to the model weights or the separate supervised training corpus. Refer to each model repository for its applicable licensing terms.

See [LICENSE](LICENSE.txt) for details.
