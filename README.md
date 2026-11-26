
# QFrBLiMa: a Quebec French Benchmark of Linguistic Markers

**QFrBLiMa** is a contrastive benchmark designed to evaluate language models' ability to recognize linguistic expressions associated with **Quebec French (QFr)** compared with **Metropolitan French (MFr)**. We evaluate both large language models (LLMs) and supervised encoder classifiers.

Unlike grammatical minimal-pair benchmarks, QFrBLiMa evaluates **regional linguistic recognition rather than grammatical acceptability**. Both sentence variants may be grammatically correct.

The benchmark comprises **1,205 contrastive sentence pairs**, covering **241 linguistic markers** across five contexts per marker. The markers are grounded in terminology documented by the *Grand dictionnaire terminologique* (GDT), maintained by the Office québécois de la langue française (OQLF).

**Dataset:** [Hugging Face](https://huggingface.co/datasets/chaouin/qfrblima) | **Paper:** Preprint forthcoming

## Terminology and Notation

The scientific notation is **MFr (Metropolitan French)** and **QFr (Quebec French)**. For backward compatibility, legacy identifiers `FI` (0) and `FQ` (1) are retained in the dataset, code, prompts and historical results. These denote the same operational categories, not a claim that International French and Metropolitan French are linguistically synonymous.

## Dataset Statistics

| Property | Value |
| --- | --- |
| Language | French (Quebec French / Metropolitan French) |
| Linguistic markers | 241 |
| Sentence pairs | 1,205 |
| Contexts per marker | 5 |
| Evaluation | Zero-shot LLM judgment and supervised encoder scoring |
| Candidate orders | Original and inverted (LLMs only) |
| Models evaluated | 8 LLMs and 5 encoder classifiers |
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

Each pair contains two sentences expressing substantially the same meaning, with a targeted regional linguistic contrast.

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
| `marker_FI` | Target MFr expression |
| `marker_FQ` | Target QFr expression |
| `source` | OQLF GDT source URL |
| `concept_id` | Concept identifier |

## Benchmark Construction

QFrBLiMa was constructed using terminology from the OQLF's *Grand dictionnaire terminologique*.

Linguistic markers were manually selected and curated from extracted terminological variants, removing redundant forms and retaining representative regional contrasts.

Five sentence contexts were manually constructed for each of the **241 retained markers**, resulting in **1,205 contrastive sentence pairs**.

The benchmark reflects documented regional linguistic associations rather than exclusive geographic usage. The constructed sentences were not independently validated by external annotators.

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

A prediction is correct when the model identifies the documented QFr candidate. `EQUIVALENT` is not counted as correct for these contrastive items.

All sentence pairs are evaluated in both candidate orders. **Order-robust accuracy** measures the proportion of pairs correctly identified in both presentations.

The evaluation prompt and output schema are available in [`qfrblima/protocol.py`](qfrblima/protocol.py).

### Encoders: independent sentence scoring

The five encoder classifiers are fine-tuned on a separate regional classification corpus. Each sentence is scored independently using the predicted probability of QFr. A pair is correct when the QFr sentence receives a strictly higher QFr probability than its MFr counterpart; ties count as incorrect. Candidate-order inversion is not required for this protocol.

Sentence-level accuracy and macro-F1 are also reported using a classification threshold of 0.5. The implementation is available in [`scripts/encoders/evaluate_classifiers.py`](scripts/encoders/evaluate_classifiers.py).

### Supervised training corpus

The separate corpus contains 938,484 examples across six genres, balanced between MFr and QFr within each genre and partition. Genre sizes differ. QFrBLiMa is used only for external evaluation, not for classifier fine-tuning or checkpoint selection.

| Partition | Examples |
| --- | ---: |
| Training | 751,288 |
| Validation | 93,768 |
| Test | 93,428 |
| Total | 938,484 |

The classifiers were trained for three epochs with seed 42, a learning rate of 2e-5, an effective batch size of 32 and a maximum sequence length of 128. Checkpoints were selected using validation macro-F1. See [`data/training_corpus/README.md`](data/training_corpus/README.md) for source provenance, construction instructions, aggregate statistics and reconstruction limitations.

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

| Model | Internal test macro-F1 | QFrBLiMa macro-F1 | Strict pairwise accuracy (95% CI) |
| --- | ---: | ---: | ---: |
| CamemBERT | 97.56% | 59.20% | 72.45% (67.97–76.68%) |
| CamemBERTv2 | 97.07% | 54.89% | 70.37% (65.73–74.85%) |
| ModernCamemBERT | 97.09% | 53.24% | 69.38% (64.98–73.61%) |
| BERT-base-cased | 95.45% | 52.44% | 58.01% (52.78–63.07%) |
| RoBERTa-base | 96.36% | 53.68% | 59.67% (54.94–64.32%) |

Internal training results, external predictions and statistical analyses are available in [`results/encoders/`](results/encoders/).

### Statistical analysis

Confidence intervals use 20,000 bootstrap replicates clustered by linguistic marker. Within-family comparisons use paired marker-level permutation tests (20,000 permutations), with Holm correction for multiple comparisons. The analysis uses seed 42.

The LLM and encoder protocols differ in their inputs and decision rules. Cross-family comparisons are descriptive rather than tests of equivalent performance measures. Under independent uniform binary guesses without ties or equivalence responses, chance accuracy is 50% for a single presentation or pairwise ordering and 25% for correctness in both LLM presentations.

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

### LLM analysis

Generate the randomized candidate presentations:

```bash
python scripts/prepare.py
```

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

These analyses do not require API keys or a GPU.

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

### Encoder training and inference

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
├── configs/                # Model configurations
├── data/
│   ├── qfrblima.csv        # Benchmark dataset
│   ├── raw/                # OQLF source data
│   ├── intermediate/       # Extracted terminology
│   ├── manifests/          # Candidate-order manifests
│   └── training_corpus/    # Supervised corpus documentation and statistics
├── qfrblima/               # Evaluation protocol
├── scripts/                # Extraction, inference, analysis
│   └── encoders/
│       ├── corpus/         # Corpus preparation utilities
│       ├── train_classifier.py
│       ├── evaluate_classifiers.py
│       ├── analyze_classifiers.py
│       └── compare_models.py
├── results/
│   ├── predictions/        # LLM predictions
│   ├── analysis/           # LLM analyses
│   └── encoders/
│       ├── training/       # Historical fine-tuning results
│       ├── evaluation/     # QFrBLiMa encoder predictions and metrics
│       ├── analysis/       # Encoder statistical analyses
│       └── comparison/     # Cross-family descriptive comparison
├── slurm/                  # Cluster execution
│   └── encoders/
├── tests/                  # Protocol tests
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

MFr is an operational comparison category, not a homogeneous linguistic variety; some supervised sources are not exclusively metropolitan. Sentence contexts were researcher-authored and were not independently validated by external annotators.

Results should therefore be interpreted as contrastive regional linguistic recognition rather than comprehensive Quebec French proficiency.

The supervised corpus uses source-based variety labels, which are imperfect proxies for regional linguistic usage. High internal classification scores do not necessarily imply strong contrastive recognition. The different evaluation protocols also limit direct comparison between LLMs and encoders.

Some historical preparation steps and source versions remain unresolved. See the data and model availability statement below.

## Data and Model Availability

Following the approach of [Khan et al. (2026)](https://aclanthology.org/2026.lrec-1.840/), we do not publicly redistribute the supervised training corpus, including its raw texts and generated training, validation and test splits, due to copyright and source-specific usage restrictions.

We provide corpus construction scripts, source documentation, aggregate corpus statistics, training configurations, evaluation pipelines and experimental results. Source-specific access and usage conditions are documented in [`data/training_corpus/sources.yaml`](data/training_corpus/sources.yaml).

The benchmark is available on [Hugging Face](https://huggingface.co/datasets/chaouin/qfrblima). We intend to release the five fine-tuned regional classifiers on Hugging Face, subject to confirmation of the applicable pretrained-model licenses and training-data usage permissions. Fine-tuned weights are not yet distributed; links will be added when available.

## Citation

If you use QFrBLiMa in your research, please cite the associated paper:

*QFrBLiMa: a Quebec French Benchmark of Linguistic Markers.*

A BibTeX citation and preprint link will be added following publication.

## License

QFrBLiMa is distributed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

The benchmark incorporates terminology from the *Grand dictionnaire terminologique* (GDT), maintained by the **Office québécois de la langue française (OQLF)** and distributed through **Données Québec**.

Original source attribution and applicable licensing conditions must be preserved. The benchmark license does not grant rights to third-party materials beyond those provided by their respective rights holders. The supervised training sources remain subject to their own licenses and access restrictions. Any future model release will specify its applicable license separately.

See [LICENSE](LICENSE.txt) for details.
