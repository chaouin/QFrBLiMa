
# QFrBLiMa: A Contrastive Benchmark for Quebec French Linguistic Marker Recognition in LLMs

**QFrBLiMa** is a contrastive benchmark designed to evaluate large language models' (LLMs) ability to recognize linguistic expressions associated with **Quebec French (FQ)** compared with **International French (FI)**.

Unlike grammatical minimal-pair benchmarks, QFrBLiMa evaluates **regional linguistic recognition rather than grammatical acceptability**. Both sentence variants may be grammatically correct.

The benchmark comprises **1,205 contrastive sentence pairs**, covering **241 linguistic markers** across five contexts per marker. The markers are grounded in terminology documented by the *Grand dictionnaire terminologique* (GDT), maintained by the Office québécois de la langue française (OQLF).

**Dataset:** [Hugging Face](https://huggingface.co/datasets/chaouin/qfrblima) | **Paper:** Preprint forthcoming

## Dataset Statistics

| Property | Value |
| --- | --- |
| Language | French (Quebec French / International French) |
| Linguistic markers | 241 |
| Sentence pairs | 1,205 |
| Contexts per marker | 5 |
| Evaluation | Zero-shot contrastive judgment |
| Candidate orders | Original and inverted |
| Models evaluated | 8 LLMs |
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

dataset = load_dataset("chaouin/QFrBLiMa", split="test")

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

| International French (FI) | Quebec French (FQ) |
| --- | --- |
| J'ai terminé le **puzzle** de mille pièces hier soir. | J'ai terminé le **casse-tête** de mille pièces hier soir. |
| Passe-moi la **clé plate** de douze millimètres. | Passe-moi la **clé ouverte** de douze millimètres. |

The dataset contains the following fields:

| Field | Description |
| --- | --- |
| `pair_id` | Unique sentence-pair identifier |
| `feature_id` | Linguistic marker identifier |
| `domaine` | Terminological domain |
| `text_FI` | International French sentence |
| `text_FQ` | Quebec French sentence |
| `label_FI` | FI label (0) |
| `label_FQ` | FQ label (1) |
| `marker_FI` | Target FI expression |
| `marker_FQ` | Target FQ expression |
| `source` | OQLF GDT source URL |
| `concept_id` | Concept identifier |

## Benchmark Construction

QFrBLiMa was constructed using terminology from the OQLF's *Grand dictionnaire terminologique*.

Linguistic markers were manually selected and curated from extracted terminological variants, removing redundant forms and retaining representative regional contrasts.

Five sentence contexts were manually constructed for each of the **241 retained markers**, resulting in **1,205 contrastive sentence pairs**.

The benchmark reflects documented regional linguistic associations rather than exclusive geographic usage. The constructed sentences were not independently validated by external annotators.

The source data and extraction code are available in [`data/`](data/) and [`scripts/extract_oqlf_markers.py`](scripts/extract_oqlf_markers.py).

## Evaluation Method

Models receive two anonymized sentence candidates (A and B) without access to the gold variety labels or expected linguistic markers.

The task is to identify which candidate exhibits stronger Quebec French linguistic characteristics, or select `EQUIVALENT`.

Five prediction labels are permitted:

- `A_MUCH_MORE_FQ`
- `A_SLIGHTLY_MORE_FQ`
- `EQUIVALENT`
- `B_SLIGHTLY_MORE_FQ`
- `B_MUCH_MORE_FQ`

A prediction is correct when the model identifies the documented FQ candidate. `EQUIVALENT` is not counted as correct for these contrastive items.

All sentence pairs are evaluated in both candidate orders. **Order-robust accuracy** measures the proportion of pairs correctly identified in both presentations.

The evaluation prompt and output schema are available in [`QFrBLiMa/protocol.py`](qfrblima/protocol.py).

## Experiments & Results

We evaluated **eight LLMs** (three API-based and five open-weight models) using both original and inverted candidate orders.

The complete predictions, evaluation results, and statistical analyses are available in [`results/`](results/).

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

Generate the randomized candidate presentations:

```bash
python scripts/prepare.py
```

Reproduce the analyses using the released model predictions:

```bash
python scripts/analyze.py
```

These analyses do not require API keys or a GPU.

For new model evaluations, see [`scripts/evaluate.py`](scripts/evaluate.py), [`configs/models.yaml`](configs/models.yaml), and [`slurm/`](slurm/).

## Repository Structure

```text
QFrBLiMa/
├── configs/                # Model configurations
├── data/
│   ├── qfrblima.csv        # Benchmark dataset
│   ├── raw/                # OQLF source data
│   ├── intermediate/       # Extracted terminology
│   └── manifests/          # Candidate-order manifests
├── qfrblima/               # Evaluation protocol
├── scripts/                # Extraction, inference, analysis
├── results/                # Predictions and analyses
├── slurm/                  # Cluster execution
├── tests/                  # Protocol tests
├── requirements.txt
├── LICENSE.txt
└── README.md
```

## Limitations

QFrBLiMa evaluates recognition of documented regional linguistic associations rather than exclusive distinctions between French varieties.

International French is an operational comparison category, not a homogeneous linguistic variety. Sentence contexts were researcher-authored and were not independently validated by external annotators.

Results should therefore be interpreted as contrastive regional linguistic recognition rather than comprehensive Quebec French proficiency.

## Citation

If you use QFrBLiMa in your research, please cite the associated paper:

*QFrBLiMa: A Contrastive Benchmark for Quebec French Linguistic Marker Recognition in LLMs.*

A BibTeX citation and preprint link will be added following publication.

## License

QFrBLiMa is distributed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

The benchmark incorporates terminology from the *Grand dictionnaire terminologique* (GDT), maintained by the **Office québécois de la langue française (OQLF)** and distributed through **Données Québec**.

Original source attribution and applicable licensing conditions must be preserved.

See [LICENSE](LICENSE.txt) for details.
