#!/usr/bin/env python3
"""Evaluate fine-tuned regional French encoders on QFrBLiMa without training.

Each candidate is scored independently (FI/MFr=0, FQ/QFr=1).
A contrast is correct when the FQ candidate has a strictly higher FQ score.
The absolute classification threshold is 0.5; score ties are incorrect
for contrastive attribution. Softmax scores are not calibrated probabilities.
The benchmark must contain 1,205 pairs covering 241 markers, five per marker.
"""
import argparse
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--pairs-csv', type=Path, required=True, help='CSV with text_FI, text_FQ')
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--bootstrap', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    return p.parse_args()

@torch.inference_mode()
def scores(texts, tokenizer, model, device, batch_size, max_length):
    result = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start+batch_size], padding=True, truncation=True,
                        max_length=max_length, return_tensors='pt')
        enc = {k: v.to(device) for k,v in enc.items()}
        prob = model(**enc).logits.float().softmax(-1)[:, 1]
        result.extend(prob.cpu().tolist())
    return np.asarray(result, dtype=float)

def boot(x, repetitions, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    if not len(x): return [None, None]
    means = np.empty(repetitions)
    for j in range(repetitions):
        means[j] = x[rng.integers(0, len(x), len(x))].mean()
    return np.percentile(means, [2.5,97.5]).tolist()

def metrics(df, bootstrap, seed):
    fi = df.score_fq_FI.to_numpy(float)
    fq = df.score_fq_FQ.to_numpy(float)
    yy = np.r_[np.zeros(len(fi), dtype=int), np.ones(len(fq), dtype=int)]
    pp = np.r_[(fi >= .5).astype(int), (fq >= .5).astype(int)]
    delta = fq - fi
    return dict(n_pairs=len(df), n_examples=2*len(df),
        accuracy=float(accuracy_score(yy, pp)),
        balanced_accuracy=float(balanced_accuracy_score(yy, pp)),
        f1_macro=float(f1_score(yy, pp, average='macro', zero_division=0)),
        confusion_matrix_FI_FQ=confusion_matrix(yy, pp, labels=[0,1]).tolist(),
        mean_FI_score_fq=float(fi.mean()), mean_FQ_score_fq=float(fq.mean()),
        median_FQ_score_fq=float(np.median(fq)),
        FQ_score_ge_05=float(np.mean(fq >= .5)), FQ_score_ge_08=float(np.mean(fq >= .8)),
        FQ_score_ge_09=float(np.mean(fq >= .9)), FI_score_lt_05=float(np.mean(fi < .5)),
        mean_paired_delta=float(delta.mean()), median_paired_delta=float(np.median(delta)),
        delta_95ci=boot(delta, bootstrap, seed),
        pairwise_FQ_higher=float(np.mean(delta>0)), pairwise_ties=float(np.mean(delta==0)),
        both_correct=float(np.mean((fi < .5)&(fq >= .5))))

def prepare_pairs(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    for target, aliases in {
        'text_FI': ('sentence_FI', 'sentence_MFr', 'text_MFr', 'sentence_fi', 'text_fi'),
        'text_FQ': ('sentence_FQ', 'sentence_QFr', 'text_QFr', 'sentence_fq', 'text_fq'),
        'feature_id': ('marker_id',),
    }.items():
        if target not in df:
            matches = [c for c in aliases if c in df]
            if len(matches) == 1:
                df[target] = df[matches[0]]
    required = {'pair_id', 'feature_id', 'text_FI', 'text_FQ'}
    if not required.issubset(df):
        raise ValueError(f'Missing {required - set(df)}; available columns: {list(df.columns)}')
    if len(df) != 1205 or df.pair_id.duplicated().any():
        raise ValueError('Expected exactly 1205 unique pair IDs')
    if any(df[c].str.strip().eq('').any() for c in required):
        raise ValueError('Empty required field')
    counts = df.groupby('feature_id').size()
    if len(counts) != 241 or not counts.eq(5).all():
        raise ValueError('Expected 241 features with exactly five contexts each')
    if df.text_FI.eq(df.text_FQ).any():
        raise ValueError('Identical candidate texts')
    return df


def main():
    a = args_parser()
    if not a.pairs_csv.is_file(): raise FileNotFoundError(a.pairs_csv)
    if not (a.model/'config.json').is_file(): raise FileNotFoundError(a.model/'config.json')
    df = prepare_pairs(a.pairs_csv)
    print(f'Validated {len(df)} pairs, {df.feature_id.nunique()} markers', flush=True)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if a.device == 'auto' else a.device
    if device == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    tok = AutoTokenizer.from_pretrained(str(a.model), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(a.model), local_files_only=True).to(device).eval()
    labels = {str(k).upper():int(v) for k,v in model.config.label2id.items()}
    if labels != {'FI':0,'FQ':1}: raise ValueError(f'Model labels not FI=0/FQ=1: {labels}')
    fi = scores(df.text_FI.tolist(), tok, model, device, a.batch_size, a.max_length)
    fq = scores(df.text_FQ.tolist(), tok, model, device, a.batch_size, a.max_length)
    df['score_fq_FI'] = fi
    df['score_fq_FQ'] = fq
    df['pred_FI'] = np.where(fi>=.5, 'FQ','FI')
    df['pred_FQ'] = np.where(fq>=.5, 'FQ','FI')
    df['delta_fq'] = fq-fi
    df['pairwise_success'] = fq>fi
    df['both_correct'] = (fi<.5)&(fq>=.5)
    a.output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.output_dir/'paired_predictions.csv', index=False)
    result = metrics(df, a.bootstrap, a.seed)
    marker = df.groupby('feature_id').agg(
        n_contexts=('pairwise_success', 'size'),
        n_correct=('pairwise_success', 'sum'),
        mean_delta=('delta_fq', 'mean'),
    ).reset_index()
    marker['recognition'] = np.select(
        [marker.n_correct.eq(marker.n_contexts), marker.n_correct.eq(0)],
        ['full', 'none'], default='partial')
    marker.to_csv(a.output_dir/'marker_recognition.csv', index=False)
    result['pairwise_marker_cluster_95ci'] = boot(
        marker.n_correct / marker.n_contexts, a.bootstrap, a.seed)
    result['delta_marker_cluster_95ci'] = boot(marker.mean_delta, a.bootstrap, a.seed)
    result['marker_recognition'] = {k: int((marker.recognition == k).sum())
                                    for k in ('full', 'partial', 'none')}
    result['benchmark_sha256'] = hashlib.sha256(a.pairs_csv.read_bytes()).hexdigest()
    result['bootstrap_replicates'] = a.bootstrap
    result['seed'] = a.seed
    result.update(checkpoint=str(a.model), input_csv=str(a.pairs_csv),
        note='Softmax scores are not calibrated probabilities; no training performed.')
    (a.output_dir/'overall_metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    for name in ('category','domaine','domain','type','marker_type','linguistic_category','phenomenon'):
        if name in df:
            rows=[]
            for key,g in df.groupby(name, dropna=False):
                m=metrics(g, min(a.bootstrap, 1000), a.seed)
                m.pop('delta_95ci', None)
                m.pop('confusion_matrix_FI_FQ')
                rows.append({name:key,**m})
            pd.DataFrame(rows).to_csv(a.output_dir/f'by_{name}.csv', index=False)
    df.sort_values('delta_fq').head(100).to_csv(a.output_dir/'worst_pairs.csv', index=False)
    print('Saved:', a.output_dir, flush=True)

if __name__ == '__main__': main()
