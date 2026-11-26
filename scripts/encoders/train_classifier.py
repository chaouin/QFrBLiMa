#!/usr/bin/env python3
"""Fine-tune a regional French sequence classifier (FI=0, FQ=1).

Supports locally available Hugging Face encoder checkpoints, including CamemBERT,
CamemBERTv2, ModernCamemBERT, BERT and RoBERTa. Requires a CUDA GPU.
Input directory: train.csv, eval.csv, test.csv; required columns: text, label,
source, genre. FI and FQ correspond to MFr and QFr in the paper.
Select the checkpoint by validation macro-F1; evaluate test only after selection.
Outputs: best_model/, history.csv, val_metrics.json, test_metrics.json,
         test_by_source.csv, test_by_genre.csv, test_predictions.csv, run_config.json.
Internal test predictions contain source texts and should not be redistributed
without verifying the applicable data rights.
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_recall_fscore_support,
                             roc_auc_score)
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)


class TextDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length):
        self.frame = frame.reset_index(drop=True)
        self.enc = tokenizer(self.frame.text.astype(str).tolist(),
                             truncation=True, max_length=max_length,
                             padding=False)
        self.labels = self.frame.label.astype(int).tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = {key: value[index] for key, value in self.enc.items()}
        x['labels'] = self.labels[index]
        return x


def read_split(path):
    frame = pd.read_csv(
        path,
        keep_default_na=False,
        dtype={
            "text": str,
            "source": str,
            "genre": str,
            "variety": str,
            "document_id": str,
        },
        low_memory=False,
    )

    required = {
        "text",
        "label",
        "source",
        "genre",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise ValueError(
            f"{path}: missing columns "
            f"{sorted(missing)}"
        )

    frame["label"] = pd.to_numeric(
        frame["label"],
        errors="raise"
    ).astype(int)

    if not frame["label"].isin(
        [0, 1]
    ).all():

        raise ValueError(
            f"{path}: invalid labels"
        )

    if frame["text"].eq("").any():

        raise ValueError(
            f"{path}: empty texts"
        )

    return frame


def score(y, pred, p_fq):
    result = {
        'n': int(len(y)),
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'f1_macro': float(f1_score(y, pred, average='macro', zero_division=0)),
        'f1_fq': float(f1_score(y, pred, pos_label=1, zero_division=0)),
        'confusion_matrix_rows_true_0_1': confusion_matrix(y, pred, labels=[0, 1]).tolist(),
        'classification_report': classification_report(
            y, pred, labels=[0, 1], target_names=['FI', 'FQ'],
            output_dict=True, zero_division=0),
    }
    if len(np.unique(y)) == 2:
        result['roc_auc'] = float(roc_auc_score(y, p_fq))
    else:
        result['roc_auc'] = None
    return result


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    probs = []
    labels = []
    for batch in loader:
        labels.extend(batch.pop('labels').numpy().tolist())
        batch = {key: val.to(device, non_blocking=True) for key, val in batch.items()}
        logits = model(**batch).logits
        probs.extend(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().tolist())
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    return y, (p >= 0.5).astype(np.int64), p


def subgroup_table(frame, pred, probs, column):
    data = frame.copy()
    data['pred_label'] = pred
    data['score_fq'] = probs
    out = []
    for name, group in data.groupby(column, dropna=False, sort=True):
        y = group.label.to_numpy(dtype=int)
        yp = group.pred_label.to_numpy(dtype=int)
        p = group.score_fq.to_numpy(dtype=float)
        report = score(y, yp, p)
        out.append({'group': name, 'n': len(group),
                    'FI': int((y == 0).sum()), 'FQ': int((y == 1).sum()),
                    'accuracy': report['accuracy'], 'f1_macro': report['f1_macro'],
                    'f1_fq': report['f1_fq'], 'roc_auc': report['roc_auc'],
                    'mean_score_fq': float(p.mean())})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--model', type=str, required=True,
                    help='Offline HF snapshot directory or model ID')
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--grad-accum', type=int, default=2)
    ap.add_argument('--eval-batch-size', type=int, default=64)
    ap.add_argument('--max-length', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--warmup-ratio', type=float, default=0.06)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1:
        ap.error('epochs, batch-size and grad-accum must be >=1')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: submit this script to a GPU Slurm node')
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'run_config.json').open('w') as f:
        json.dump({**vars(args), 'data_dir': str(args.data_dir),
                   'output_dir': str(args.output_dir),
                   'torch': torch.__version__}, f, indent=2)

    train = read_split(args.data_dir / "train.csv")
    val = read_split(args.data_dir / "eval.csv")
    test = read_split(args.data_dir / "test.csv")
    # All splits must be disjoint after the final cleanup.
    key_sets = [{(x.casefold().strip()) for x in d.text} for d in (train, val, test)]
    if any(key_sets[i] & key_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Exact text overlap detected between train/val/test')
    print('Label counts:', {name: d.label.value_counts().to_dict() for name,d in
                             [('train',train),('val',val),('test',test)]}, flush=True)
    print('Sizes:', len(train), len(val), len(test), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True,
                                              use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=2, id2label={0: 'FI', 1: 'FQ'},
        label2id={'FI': 0, 'FQ': 1}, local_files_only=True,
        ignore_mismatched_sizes=True).to(device)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    train_ds = TextDataset(train, tokenizer, args.max_length)
    val_ds = TextDataset(val, tokenizer, args.max_length)
    test_ds = TextDataset(test, tokenizer, args.max_length)
    g = torch.Generator().manual_seed(args.seed)
    common = dict(collate_fn=collator, num_workers=args.workers,
                  pin_memory=True, persistent_workers=args.workers > 0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size,
                            shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size,
                             shuffle=False, **common)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    updates_per_epoch = (len(train_loader) + args.grad_accum - 1) // args.grad_accum
    steps = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, int(steps * args.warmup_ratio)),
        num_training_steps=steps)
    use_amp = True
    scaler = torch.amp.GradScaler('cuda')
    history = []
    best = -1.0
    best_epoch = None
    best_dir = args.output_dir / 'best_model'
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            # Scale the last, possibly smaller accumulation window correctly.
            window = min(args.grad_accum,
                         len(train_loader) - (i // args.grad_accum) * args.grad_accum)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            running_loss += loss.item()
            scaler.scale(loss / window).backward()
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            if (i + 1) % 500 == 0:
                print(f'epoch={epoch} batches={i+1}/{len(train_loader)} '
                      f'loss={running_loss/(i+1):.4f}', flush=True)
        y, yp, p = predict(model, val_loader, device)
        metrics = score(y, yp, p)
        row = {'epoch': epoch, 'train_loss': running_loss / len(train_loader),
               'val_accuracy': metrics['accuracy'], 'val_f1_macro': metrics['f1_macro'],
               'val_roc_auc': metrics['roc_auc']}
        history.append(row)
        pd.DataFrame(history).to_csv(args.output_dir / 'history.csv', index=False)
        print('VALIDATION:', row, flush=True)
        if metrics['f1_macro'] > best:
            best = metrics['f1_macro']
            best_epoch = epoch
            model.save_pretrained(best_dir, safe_serialization=True)
            tokenizer.save_pretrained(best_dir)
            with (args.output_dir / 'val_metrics.json').open('w') as f:
                json.dump({'best_epoch': best_epoch, **metrics}, f, indent=2)
    print(f'Reload best model epoch={best_epoch} val_f1={best:.5f}', flush=True)
    best_model = AutoModelForSequenceClassification.from_pretrained(
        best_dir, local_files_only=True).to(device)
    y, yp, p = predict(best_model, test_loader, device)
    metrics = score(y, yp, p)
    with (args.output_dir / 'test_metrics.json').open('w') as f:
        json.dump({'best_epoch': best_epoch, **metrics}, f, indent=2)
    subgroup_table(test, yp, p, 'source').to_csv(
        args.output_dir / 'test_by_source.csv', index=False)
    subgroup_table(
        test,
        yp,
        p,
        "genre"
    ).to_csv(
        args.output_dir /
        "test_by_genre.csv",
        index=False
    )
    pred_frame = test.copy()
    pred_frame['pred_label'] = yp
    pred_frame['score_fq'] = p  # uncalibrated softmax, not a calibrated probability
    pred_frame.to_csv(args.output_dir / 'test_predictions.csv', index=False)
    print('TEST:', {k: v for k, v in metrics.items() if k != 'classification_report'},
          flush=True)
    print('Saved:', args.output_dir, flush=True)

if __name__ == '__main__':
    main()
