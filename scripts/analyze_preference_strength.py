#!/usr/bin/env python3
"""Run from repository root with script installed in scripts/. No inference."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import MODELS, LABELS, load_benchmark, read_valid

ENCODERS={'camembert':'CamemBERT','camembertv2':'CamemBERTv2','moderncamembert':'ModernCamemBERT','roberta_base':'RoBERTa-base','bert_base_cased':'BERT-base-cased'}
CATS=['strong_correct','slight_correct','no_preference','slight_incorrect','strong_incorrect']

def summarize(model,strategy,values,**kwargs):
    counts=pd.Series(values).value_counts().reindex(CATS,fill_value=0)
    return dict(model=model,strategy=strategy,n=len(values),**kwargs,**{c:int(counts[c]) for c in CATS},**{c+'_pct':100*counts[c]/len(values) for c in CATS})

def bins(d,low,high):
    return np.select([d>=high,d>=low,d<=-high,d<=-low],[CATS[0],CATS[1],CATS[4],CATS[3]],default=CATS[2])

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,default=Path('results/analysis'));a=p.parse_args();r=a.root;out=a.out;out.mkdir(parents=True,exist_ok=True)
    b=load_benchmark(r/'data/qfrblima.csv',strict=True).reset_index(drop=True)
    rows=[];sens=[];diag=[]
    for m in MODELS:
        o,i=[read_valid(r/'results/predictions'/m,None,b,order) for order in ['original','inverted']]
        if not (i.true_fq_position.eq(o.true_fq_position.map({'A':'B','B':'A'})).all() and i.candidate_a.eq(o.candidate_b).all() and i.candidate_b.eq(o.candidate_a).all() and i.original_judgment_id.eq(o.judgment_id).all()): raise ValueError(m+' inversion mismatch')
        d=pd.concat([o,i]);strong=d.overall_preference.str.contains('_MUCH_');eq=d.overall_preference.eq('EQUIVALENT');correct=d.correct_calc
        vals=np.select([eq,strong&correct,correct,strong],[CATS[2],CATS[0],CATS[1],CATS[4]],default=CATS[3]);rows.append(summarize(LABELS[m],'LLM',vals))
    for m,label in ENCODERS.items():
        d=pd.read_csv(r/'results/encoders/evaluation'/m/'paired_predictions.csv',keep_default_na=False)
        if not d.pair_id.is_unique or set(d.pair_id)!=set(b.pair_id): raise ValueError(m+' pair IDs')
        d=d.set_index('pair_id').loc[b.pair_id].reset_index()
        for c in ['feature_id','text_FI','text_FQ']:
            if not d[c].eq(b[c]).all(): raise ValueError(m+' '+c)
        scores=d[['score_fq_FI','score_fq_FQ']].to_numpy(float);delta=scores[:,1]-scores[:,0]
        if not np.isfinite(scores).all() or not ((scores>=0)&(scores<=1)).all() or not np.allclose(delta,d.delta_fq.astype(float),atol=1e-12,rtol=0): raise ValueError(m+' scores')
        if not np.array_equal(delta>0,d.pairwise_success.astype(str).str.lower().eq('true')): raise ValueError(m+' ranking')
        diag.append(dict(model=label,strict_accuracy=100*np.mean(delta>0),exact_ties=int(sum(delta==0)),**{f'abs_delta_q{q}':np.percentile(abs(delta),q) for q in [10,25,50,75,90]}))
        rows.append(summarize(label,'Encoder',bins(delta,.05,.20)))
        for low,high in [(l,h) for l in [.01,.025,.05,.075,.10] for h in [.10,.15,.20,.25,.30] if l<h]: sens.append(summarize(label,'Encoder',bins(delta,low,high),low=low,high=high))
    df=pd.DataFrame(rows);df.to_csv(out/'preference_strength.csv',index=False);pd.DataFrame(sens).to_csv(out/'preference_threshold_sensitivity.csv',index=False);pd.DataFrame(diag).to_csv(out/'encoder_margin_distribution.csv',index=False)
    print(df[['model']+[c+'_pct' for c in CATS]].round(2).to_string(index=False));print(pd.DataFrame(diag).round(4).to_string(index=False))
if __name__=='__main__':main()
