#!/usr/bin/env bash
#SBATCH --job-name=qfr_train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
# Optional site-specific environment setup (modules, virtualenv).
if [[ -n "${QFR_ENV_SETUP:-}" ]]; then
    source "$QFR_ENV_SETUP"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python -u - <<'PYCODE'
import os
import subprocess
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path("configs/encoders.yaml").read_text())
names = ["camembert", "camembertv2", "moderncamembert", "bert_base_cased", "roberta_base"]
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
if not 0 <= idx < len(names):
    raise ValueError(f"Invalid array index: {idx}")
name = names[idx]
model_cfg = config["models"][name]
seed = config["seed"]
root = Path(os.environ.get("QFR_CHECKPOINT_ROOT", "checkpoints"))
run = root / name

def require(path):
    if not path.is_file():
        raise FileNotFoundError(path)

def flags(mapping, section):
    return [str(v) for key, flag in mapping.items() for v in (flag, section[key])]

section = config["training"]
if section["selection_metric"] != "macro_f1":
    raise ValueError("Training script selects validation macro-F1 only")
base = Path(os.environ["QFR_PRETRAINED_ROOT"]) / model_cfg.get("local_dir", "camembert-base" if name == "camembert" else name)
require(base / "config.json")
data = Path(os.environ["QFR_CORPUS_ROOT"]) / "data/fq_fi_classifier_corpus"
for split in ("train", "eval", "test"):
    require(data / f"{split}.csv")
if run.exists() and any(run.iterdir()):
    raise FileExistsError(f"Refusing to overwrite existing run: {run}")
cmd = [sys.executable, "-u", "scripts/encoders/train_classifier.py",
       "--data-dir", str(data), "--model", str(base), "--output-dir", str(run),
       "--seed", str(seed), "--workers", str(section.get("workers", 4))]
cmd += flags({"epochs":"--epochs", "batch_size":"--batch-size",
              "gradient_accumulation":"--grad-accum", "eval_batch_size":"--eval-batch-size",
              "max_length":"--max-length", "learning_rate":"--lr",
              "weight_decay":"--weight-decay", "warmup_ratio":"--warmup-ratio"}, section)

print("Model:", name, flush=True)
print("Command:", cmd, flush=True)
subprocess.run(cmd, check=True)
PYCODE
