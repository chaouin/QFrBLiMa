#!/usr/bin/env python3
"""Analyze released encoder predictions; no checkpoints or inference required.

Primary endpoint: strict pairwise ordering (ties count as failures).
Inference uses marker-cluster bootstrap and paired marker-level sign flips.
Holm correction covers the ten pairwise model comparisons, two-sided.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

MODELS = {'camembert': 'CamemBERT', 'camembertv2': 'CamemBERTv2',
          'moderncamembert': 'ModernCamemBERT', 'bert_base_cased': 'BERT-base-cased',
          'roberta_base': 'RoBERTa-base'}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def resample(x, repetitions, rng, permutation=False):
    """Process in chunks to avoid large temporary arrays."""
    result = []
    for start in range(0, repetitions, 512):
        n = min(512, repetitions - start)
        if permutation:
            signs = 2 * rng.integers(0, 2, size=(n, len(x))) - 1
            result.extend((signs * x).mean(axis=1))
        else:
            idx = rng.integers(0, len(x), size=(n, len(x)))
            result.extend(x[idx].mean(axis=1))
    return np.asarray(result)


def ci(x, n, seed):
    return np.percentile(resample(x, n, np.random.default_rng(seed)), [2.5, 97.5])


def save(fig, out, name):
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'{name}.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', type=Path, default=Path('results/encoders/evaluation'))
    p.add_argument('--training', type=Path, default=Path('results/encoders/training'))
    p.add_argument('--benchmark', type=Path, default=Path('data/qfrblima.csv'))
    p.add_argument('--out', type=Path, default=Path('results/encoders/analysis'))
    p.add_argument('--bootstrap', type=int, default=20000)
    p.add_argument('--permutations', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    require(a.bootstrap > 0 and a.permutations > 0, 'Replicate counts must be positive')
    digest = hashlib.sha256(a.benchmark.read_bytes()).hexdigest()
    benchmark = pd.read_csv(a.benchmark, dtype=str, keep_default_na=False).set_index('pair_id').sort_index()
    require(len(benchmark) == 1205 and benchmark.index.is_unique, 'Invalid benchmark pair IDs')
    marker_scores, rows, cms, domains = {}, [], {}, []
    metadata = None
    for model, name in MODELS.items():
        folder = a.evaluation / model
        m = read_json(folder / 'overall_metrics.json')
        require(m['benchmark_sha256'] == digest, f'{model}: benchmark hash mismatch')
        df = pd.read_csv(folder / 'paired_predictions.csv', dtype={'pair_id': str, 'feature_id': str}, keep_default_na=False).set_index('pair_id').sort_index()
        require(df.index.equals(benchmark.index), f'{model}: pair IDs mismatch')
        cols = ['feature_id', 'text_FI', 'text_FQ', 'domaine']
        for col in cols:
            if col in benchmark:
                require(df[col].astype(str).equals(benchmark[col]), f'{model}: benchmark {col} mismatch')
        if metadata is None:
            metadata = df[cols].copy()
        else:
            require(df[cols].equals(metadata), f'{model}: inconsistent pair metadata')
        counts = df.groupby('feature_id').size()
        require(len(counts) == 241 and counts.eq(5).all(), f'{model}: invalid marker clusters')
        fi, fq = df.score_fq_FI.to_numpy(float), df.score_fq_FQ.to_numpy(float)
        require(np.isfinite(fi).all() and np.isfinite(fq).all() and ((fi >= 0)&(fi <= 1)&(fq >= 0)&(fq <= 1)).all(), f'{model}: invalid scores')
        y = np.repeat([0, 1], len(df)); pred = np.r_[fi >= .5, fq >= .5].astype(int)
        cm = confusion_matrix(y, pred, labels=[0, 1]); cms[model] = cm
        df['success'] = fq > fi
        marker = df.groupby('feature_id').success.mean().sort_index()
        marker_scores[model] = marker
        low, high = ci(marker.to_numpy(), a.bootstrap, a.seed)
        val = read_json(a.training / model / 'val_metrics.json')
        test = read_json(a.training / model / 'test_metrics.json')
        f1 = f1_score(y, pred, average='macro')
        row = dict(model=model, name=name, internal_val_f1=val['f1_macro'], internal_test_f1=test['f1_macro'], best_epoch=val['best_epoch'],
                   accuracy=(y == pred).mean(), f1_macro=f1, pairwise_accuracy=marker.mean(), ci_low=low, ci_high=high,
                   both_correct=((fi < .5)&(fq >= .5)).mean(), ties=(fi == fq).mean(),
                   recall_MFr=cm[0,0]/cm[0].sum(), recall_QFr=cm[1,1]/cm[1].sum(),
                   full=int(marker.eq(1).sum()), partial=int(((marker > 0)&(marker < 1)).sum()), none=int(marker.eq(0).sum()),
                   f1_drop=test['f1_macro']-f1)
        for key, value in [('accuracy', row['accuracy']), ('f1_macro', f1), ('pairwise_FQ_higher', marker.mean()), ('both_correct', row['both_correct']), ('pairwise_ties', row['ties'])]:
            require(np.isclose(m[key], value, atol=1e-10, rtol=0), f'{model}: historical {key} mismatch')
        require(np.array_equal(cm, m['confusion_matrix_FI_FQ']), f'{model}: historical confusion matrix mismatch')
        rows.append(row)
        for domain, group in df.groupby('domaine', dropna=False):
            domains.append(dict(model=model, domaine=domain, n_pairs=len(group), n_markers=group.feature_id.nunique(), pairwise_accuracy=group.success.mean()))
    summary = pd.DataFrame(rows)
    scores = pd.DataFrame(marker_scores)
    comparisons = []
    for first, second in itertools.combinations(MODELS, 2):
        d = (scores[first]-scores[second]).to_numpy()
        observed = d.mean()
        null = resample(d, a.permutations, np.random.default_rng(a.seed), permutation=True)
        raw = (1 + np.count_nonzero(np.abs(null) >= abs(observed)-1e-12))/(a.permutations+1)
        low, high = ci(d, a.bootstrap, a.seed)
        comparisons.append(dict(model_a=first, model_b=second, delta_a_minus_b=observed, ci_low=low, ci_high=high, p_raw=raw))
    comp = pd.DataFrame(comparisons)
    comp['p_holm'] = holm(comp.p_raw)
    comp['significant_005'] = comp.p_holm < .05
    a.out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(a.out/'encoder_summary.csv', index=False)
    scores.to_csv(a.out/'marker_pairwise_accuracy.csv')
    comp.to_csv(a.out/'pairwise_comparisons.csv', index=False)
    pd.DataFrame(domains).to_csv(a.out/'domain_accuracy.csv', index=False)
    matrix = pd.DataFrame(1., index=MODELS, columns=MODELS)
    for r in comp.itertuples():
        matrix.loc[r.model_a,r.model_b] = matrix.loc[r.model_b,r.model_a] = r.p_holm
    matrix.to_csv(a.out/'pairwise_p_holm_matrix.csv')
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.errorbar(x, summary.pairwise_accuracy*100, yerr=np.vstack([summary.pairwise_accuracy-summary.ci_low, summary.ci_high-summary.pairwise_accuracy])*100, fmt='o', capsize=4)
    ax.axhline(50, ls='--', label='Random ordering (no ties)')
    ax.set(xticks=x, xticklabels=summary.name, ylabel='Pairwise accuracy (%)', ylim=(0,100)); ax.tick_params(axis='x', rotation=20); ax.legend()
    save(fig,a.out,'pairwise_accuracy_ci')
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(x-.19, summary.internal_test_f1*100, .38, label='Internal test')
    ax.bar(x+.19, summary.f1_macro*100, .38, label='QFrBLiMa')
    ax.set(xticks=x, xticklabels=summary.name, ylabel='Macro-F1 (%)', ylim=(0,100)); ax.tick_params(axis='x', rotation=20); ax.legend()
    save(fig,a.out,'internal_external_f1')
    fig, ax = plt.subplots(figsize=(8,4.5))
    bottom = np.zeros(len(summary))
    for col in ['full','partial','none']:
        ax.bar(x, summary[col], bottom=bottom, label=col.capitalize()); bottom += summary[col].to_numpy()
    ax.set(xticks=x, xticklabels=summary.name, ylabel='Number of markers'); ax.tick_params(axis='x', rotation=20); ax.legend()
    save(fig,a.out,'marker_recognition')
    for model, cm in cms.items():
        fig, ax = plt.subplots(figsize=(4,3.5)); normalized = cm/cm.sum(axis=1,keepdims=True)
        im = ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
        for i,j in itertools.product(range(2),repeat=2):
            ax.text(j,i,f'{cm[i,j]}\n{normalized[i,j]:.1%}',ha='center',va='center',color='white' if normalized[i,j]>.55 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['MFr','QFr'],yticklabels=['MFr','QFr'],xlabel='Predicted',ylabel='True',title=MODELS[model]); fig.colorbar(im,ax=ax)
        save(fig,a.out,f'confusion_{model}')
    fig, ax = plt.subplots(figsize=(6.5,5)); im = ax.imshow(-np.log10(matrix.to_numpy()),cmap='viridis')
    for i,j in itertools.product(range(len(MODELS)),repeat=2):
        ax.text(j,i,'—' if i==j else f'{matrix.iloc[i,j]:.3g}',ha='center',va='center',color='white' if -np.log10(matrix.iloc[i,j])<2 else 'black',fontsize=8)
    ax.set(xticks=x,yticks=x,xticklabels=summary.name,yticklabels=summary.name,title='Holm-adjusted p-values'); ax.tick_params(axis='x',rotation=40); fig.colorbar(im,ax=ax,label='−log10(p)')
    save(fig,a.out,'pairwise_significance')
    (a.out/'analysis_config.json').write_text(json.dumps(dict(benchmark_sha256=digest, bootstrap=a.bootstrap, permutations=a.permutations, seed=a.seed, cluster='feature_id', endpoint='strict pairwise ordering', correction='Holm, 10 two-sided comparisons', ci='pointwise percentile marker bootstrap', permutation='paired marker-level sign flip; exchangeability assumed'),indent=2)+'\n')
    print(summary[['name','internal_test_f1','f1_macro','pairwise_accuracy']].to_string(index=False))
    print(comp.to_string(index=False)); print('Saved:', a.out)


if __name__ == '__main__':
    main()
