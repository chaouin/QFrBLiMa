#!/usr/bin/env python3
"""Apply public QFrMark IDs to historical result CSV files without changing any predictions.

The model's judgment_id, candidate contents, preferences and correctness are not modified.
The mapping only renames pair_id and feature_id. Input is never overwritten.
"""
from pathlib import Path
import argparse
import pandas as pd
p=argparse.ArgumentParser()
p.add_argument('--input',required=True,type=Path)
p.add_argument('--output',required=True,type=Path)
p.add_argument('--mapping',type=Path,default=Path('data/id_mapping.csv'))
a=p.parse_args()
if a.input.resolve()==a.output.resolve():p.error('output must differ from input')
m=pd.read_csv(a.mapping,dtype=str,keep_default_na=False)
d=pd.read_csv(a.input,dtype=str,keep_default_na=False)
lookup=dict(zip(m.old_pair_id,m.new_pair_id)); fmap=dict(zip(m.old_feature_id,m.new_feature_id))
if 'pair_id' not in d.columns:p.error('input missing pair_id')
unknown=sorted(set(d.pair_id)-set(lookup)-set(lookup.values()))
if unknown:p.error(f'unknown pair IDs: {unknown[:5]}')
d['pair_id']=d.pair_id.map(lambda x:lookup.get(x,x))
if 'feature_id' in d.columns:
 d['feature_id']=d.feature_id.map(lambda x:fmap.get(x,x))
if d.pair_id.duplicated().any():p.error('duplicate pair IDs after mapping')
a.output.parent.mkdir(parents=True,exist_ok=True)
d.to_csv(a.output,index=False)
print(f'Remapped {len(d)} rows to {a.output}')
