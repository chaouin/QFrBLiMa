#!/usr/bin/env bash
#SBATCH --job-name=qfrblima_open
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-4%4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the QFrBLiMa repository root}"
# Supply your site's allocation: sbatch --account=YOUR_ACCOUNT slurm/open_models.sbatch
# Set MODEL_ROOT to a folder with locally cached model checkpoints.
: "${MODEL_ROOT:?Set MODEL_ROOT to your local models directory}"
: "${VENV_PATH:?Set VENV_PATH to your activated Python environment path}"
mkdir -p logs
names=(llama31_8b aya_expanse_8b qwen3_8b mistral_nemo_12b gemma4_12b_it)
folders=(Llama-3.1-8B-Instruct aya-expanse-8B Qwen3-8B Mistral-Nemo-Instruct-2407 gemma-4-12B-it)
idx="${SLURM_ARRAY_TASK_ID:?SBATCH array required}"
source "$VENV_PATH/bin/activate"
python scripts/evaluate.py --benchmark data/qfrblima.csv \
  --model "${names[$idx]}" --checkpoint "$MODEL_ROOT/${folders[$idx]}" \
  --order both --offline --out results/predictions
