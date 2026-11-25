#!/usr/bin/env python3
"""Validate dataset; export the exact fixed candidate placements for both orders."""
import argparse,json,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from qfrblima.protocol import load_benchmark, make_jobs, digest
p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,default=Path('data/qfrblima.csv'))
p.add_argument('--out',type=Path,default=Path('data/manifests'));p.add_argument('--seed',type=int,default=42)
a=p.parse_args();df=load_benchmark(a.benchmark);original,inverted=make_jobs(df,a.seed)
a.out.mkdir(parents=True,exist_ok=True)
cols=['pair_id','feature_id','judgment_id','candidate_a','candidate_b','true_fq_position','original_judgment_id']
for name,rows in [('original',original),('inverted',inverted)]:
    pd.DataFrame(rows)[cols].to_csv(a.out/f'{name}.csv',index=False)
meta={'benchmark_sha256':digest(a.benchmark),'seed':a.seed,'n_pairs':len(df),'n_markers':df.feature_id.nunique(),'original_fq_a':sum(j['true_fq_position']=='A' for j in original)}
(a.out/'manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print(meta)
