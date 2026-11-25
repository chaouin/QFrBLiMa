Use `mkdir -p logs` before `sbatch`. Example:

```bash
export MODEL_ROOT="$HOME/models"
export VENV_PATH="$HOME/qfrblima-venv"
mkdir -p logs
sbatch --account=YOUR_ALLOCATION slurm/open_models.sh
```

If Gemma requires a different `transformers` environment on your installation, submit it separately with `--array=4` and its own `VENV_PATH`. Never assume the original cluster account or checkpoint directories apply to other institutions. To run only inversions, change `--order both` to `--order inverted`.
