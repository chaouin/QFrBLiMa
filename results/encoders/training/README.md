# Historical encoder fine-tuning results

These files report the original supervised training runs, not new experiments. FI (0) and FQ (1) correspond to MFr and QFr in the paper. Validation and test sets contain 93,768 and 93,428 sentences, respectively.

All five models use the same recorded hyperparameters and PyTorch 2.7.1. Checkpoints were selected by validation macro-F1. ModernCamemBERT selected epoch 2; all other models selected epoch 3.

The five metrics and history files per model are unchanged. Only the three path fields (`data_dir`, `model`, `output_dir`) in each `run_config.json` have been replaced with portable placeholders; these are archival records, not executable configuration files. No model weights or individual training-corpus texts are included.

Source-level metrics describe single-variety groups and should not be interpreted as balanced two-class evaluations. The CN2i test subset contains only six sentences; source-level accuracy for that subset is not a robust estimate. The internal test set shares source categories with training; high internal scores do not establish generalization to unseen sources or contrastive linguistic markers.
