#!/usr/bin/env bash
#SBATCH --job-name=qfr_evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:a100_3g.20gb:1
#SBATCH --time=01:00:00
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

section = config["evaluation"]
if float(section["classification_threshold"]) != 0.5:
    raise ValueError("Evaluation script uses a fixed threshold of 0.5")
# Optional historical checkpoint layout, e.g. {model}_fqfi_genre_balanced_seed42.
run = root / os.environ.get("QFR_RUN_PATTERN", "{model}").format(model=name, seed=seed)
checkpoint = run / "best_model"
require(checkpoint / "config.json")
benchmark = Path(os.environ.get("QFR_BENCHMARK", "data/qfrblima.csv"))
require(benchmark)
out = Path(os.environ.get("QFR_EVAL_ROOT", "results/encoders/evaluation")) / name
if out.exists() and any(out.iterdir()):
    raise FileExistsError(f"Refusing to overwrite existing evaluation: {out}")
cmd = [sys.executable, "-u", "scripts/encoders/evaluate_classifiers.py",
       "--pairs-csv", str(benchmark), "--model", str(checkpoint),
       "--output-dir", str(out), "--seed", str(seed), "--device", "cuda"]
cmd += flags({"batch_size":"--batch-size", "max_length":"--max-length", "bootstrap":"--bootstrap"}, section)

print("Model:", name, flush=True)
print("Command:", cmd, flush=True)
subprocess.run(cmd, check=True)
PYCODE
