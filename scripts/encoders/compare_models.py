#!/usr/bin/env python3
"""Compare LLM and encoder results descriptively on aligned QFrBLiMa pairs.

LLM mean single-order accuracy averages two presentations; robust accuracy
requires both correct. Encoder strict ordering compares independent QFr scores.
These are different protocols, so no cross-family significance tests are run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LLMS = {
    'gemini': 'Gemini 3.8 Flash',
    'claude': 'Claude Opus 4.8',
    'gpt54': 'GPT-5.4',
    'gemma4_12b_it': 'Gemma 4 12B IT',
    'mistral_nemo_12b': 'Mistral NeMo 12B Instruct',
    'qwen3_8b': 'Qwen3 8B',
    'aya_expanse_8b': 'Aya Expanse 8B',
    'llama31_8b': 'Llama 3.1 8B Instruct',
}
ENCODERS = {'camembert':'CamemBERT', 'camembertv2':'CamemBERTv2',
            'moderncamembert':'ModernCamemBERT', 'bert_base_cased':'BERT-base-cased', 'roberta_base':'RoBERTa-base'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False).set_index('pair_id').sort_index()
    require(df.index.is_unique, f'{path}: duplicate pair IDs')
    return df


def align(df, benchmark, label):
    require(df.index.equals(benchmark.index), f'{label}: pair IDs differ')
    for col in ('feature_id', 'domaine', 'marker_FI', 'marker_FQ'):
        require(df[col].equals(benchmark[col]), f'{label}: {col} differs')


def bool_col(df, col):
    s = df[col].str.lower()
    require(s.isin(['true','false']).all(), f'Invalid boolean in {col}')
    return s.eq('true')


def interval(values, n, seed):
    rng = np.random.default_rng(seed)
    samples = []
    for start in range(0,n,512):
        idx = rng.integers(0,len(values),size=(min(512,n-start),len(values)))
        samples.extend(values[idx].mean(axis=1))
    return np.percentile(samples,[2.5,97.5])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark', type=Path, default=Path('data/qfrblima.csv'))
    p.add_argument('--predictions', type=Path, default=Path('results/predictions'))
    p.add_argument('--evaluation', type=Path, default=Path('results/encoders/evaluation'))
    p.add_argument('--out', type=Path, default=Path('results/encoders/comparison'))
    p.add_argument('--bootstrap', type=int, default=20000)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    require(a.bootstrap > 0, 'Bootstrap count must be positive')
    b = load(a.benchmark)
    require(len(b)==1205 and b.groupby('feature_id').size().eq(5).all() and b.feature_id.nunique()==241, 'Unexpected benchmark size')
    digest = hashlib.sha256(a.benchmark.read_bytes()).hexdigest()
    summary, markers, domains, pairs = [], [], [], []
    def record(model, name, family, metric, score):
        data = b[['feature_id','domaine']].assign(score=score.astype(float))
        require(data.score.notna().all(), f'{model}: missing scores')
        m = data.groupby('feature_id').score.mean()
        low, high = interval(m.to_numpy(), a.bootstrap, a.seed)
        meta = dict(model=model,name=name,family=family,metric=metric)
        summary.append(dict(**meta,n_pairs=len(data),accuracy=data.score.mean(),ci_low=low,ci_high=high))
        markers.append(m.rename('accuracy').reset_index().assign(**meta))
        domains.append(data.groupby('domaine').agg(accuracy=('score','mean'),n_pairs=('score','size'),n_markers=('feature_id','nunique')).reset_index().assign(**meta))
        pairs.append(data.reset_index().assign(**meta))
    for model,name in LLMS.items():
        orders=[]
        for order in ('original','inverted'):
            d=load(a.predictions/model/order/'results.csv'); align(d,b,f'{model}/{order}')
            truth=d.true_fq_position
            require(truth.isin(['A','B']).all(), f'{model}: invalid truth position')
            require(d.candidate_a.equals(b.text_FQ.where(truth.eq('A'),b.text_FI)) and d.candidate_b.equals(b.text_FQ.where(truth.eq('B'),b.text_FI)),f'{model}/{order}: candidate mismatch')
            valid=bool_col(d,'valid'); pred=d.predicted_fq_position
            require(pred[valid].isin(['A','B','EQUIVALENT']).all(),f'{model}: invalid prediction')
            correct=valid & pred.eq(truth)
            require(correct.equals(bool_col(d,'correct')),f'{model}/{order}: correct flag mismatch')
            orders.append((truth,correct,d))
        (t1,c1,d1),(t2,c2,d2)=orders
        require(t1.ne(t2).all() and d1.candidate_a.equals(d2.candidate_b) and d1.candidate_b.equals(d2.candidate_a),f'{model}: orders not inverted')
        robust=c1 & c2
        if 'correct_both_orders' in d2:
            require(robust.equals(bool_col(d2,'correct_both_orders')),f'{model}: robust flag mismatch')
        record(model,name,'LLM','original_accuracy',c1)
        record(model,name,'LLM','inverted_accuracy',c2)
        record(model,name,'LLM','mean_single_order_accuracy',(c1.astype(float)+c2.astype(float))/2)
        record(model,name,'LLM','robust_accuracy',robust)
    for model,name in ENCODERS.items():
        folder=a.evaluation/model
        m=json.loads((folder/'overall_metrics.json').read_text())
        require(m['benchmark_sha256']==digest,f'{model}: benchmark SHA-256 mismatch')
        d=load(folder/'paired_predictions.csv'); align(d,b,model)
        for col in ('text_FI','text_FQ'):
            require(d[col].equals(b[col]),f'{model}: {col} mismatch')
        fi=d.score_fq_FI.astype(float); fq=d.score_fq_FQ.astype(float)
        require(fi.between(0,1).all() and fq.between(0,1).all(),f'{model}: invalid scores')
        correct=fq.gt(fi)
        require(correct.equals(bool_col(d,'pairwise_success')) and np.isclose(correct.mean(),m['pairwise_FQ_higher'],atol=1e-10,rtol=0),f'{model}: ordering mismatch')
        record(model,name,'Encoder','strict_pairwise_accuracy',correct)
    a.out.mkdir(parents=True,exist_ok=True)
    s=pd.DataFrame(summary)
    s.to_csv(a.out/'model_comparison.csv',index=False)
    pd.concat(markers,ignore_index=True).to_csv(a.out/'marker_comparison.csv',index=False)
    pd.concat(domains,ignore_index=True).to_csv(a.out/'domain_comparison.csv',index=False)
    pd.concat(pairs,ignore_index=True).to_csv(a.out/'pair_comparison.csv',index=False)
    fig, axes = plt.subplots(1,2,figsize=(13,5.5),sharex=True,gridspec_kw={'width_ratios':[1.35,1]})
    for ax, models, metrics, title in [(axes[0],LLMS,['mean_single_order_accuracy','robust_accuracy'],'LLMs: two presentations'),(axes[1],ENCODERS,['strict_pairwise_accuracy'],'Encoders: independent scores')]:
        y=np.arange(len(models))
        for i,metric in enumerate(metrics):
            v=s[s.metric.eq(metric)].set_index('model').loc[list(models)]
            offset=(i-(len(metrics)-1)/2)*.28
            ax.errorbar(v.accuracy*100,y+offset,xerr=np.maximum(0,np.vstack([v.accuracy-v.ci_low,v.ci_high-v.accuracy]))*100,fmt='o',capsize=3,label=metric.replace('_',' '))
        ax.set(yticks=y,yticklabels=list(models.values()),xlim=(0,100),xlabel='Accuracy (%)',title=title)
        ax.invert_yaxis();ax.legend(loc='lower right',fontsize=8);ax.grid(axis='x',alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(a.out/f'model_comparison.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    config=dict(benchmark_sha256=digest,bootstrap=a.bootstrap,seed=a.seed,cluster='feature_id',ci='pointwise percentile marker bootstrap',cross_family_tests=False,notes='Different protocols; descriptive comparison only. Equivalent, invalid and score ties count as failures. LLM mean averages two binary outcomes; robust requires both. Encoder scores are independently computed.')
    (a.out/'comparison_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(s[s.metric.isin(['mean_single_order_accuracy','robust_accuracy','strict_pairwise_accuracy'])][['name','metric','accuracy','ci_low','ci_high']].to_string(index=False))
    print('Saved:',a.out)


if __name__=='__main__':
    main()
