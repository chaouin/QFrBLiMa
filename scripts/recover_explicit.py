#!/usr/bin/env python3
"""Conservatively restore explicit preferences from failed model attempts.

Never changes original results. It only recovers a preference if every saved
attempt for an item states the same unique, explicit preference and judgment_id.
Other incomplete linguistic fields remain unchanged/blank and are audited.
No reference label is ever used to CHOOSE a recovered preference.
"""
import argparse,json,re
from pathlib import Path
import pandas as pd

PREFS={'A_MUCH_MORE_FQ','A_SLIGHTLY_MORE_FQ','EQUIVALENT','B_SLIGHTLY_MORE_FQ','B_MUCH_MORE_FQ'}
ALIASES={'EQUIVALENCE':'EQUIVALENT','EQUIVALENTEQUIVALENT':'EQUIVALENT'}
JID=re.compile(r'''["']judgment_id["']\s*:\s*["']([^"']+)["']''')
PREF=re.compile(r'''["']overall_preference["']\s*:\s*["']?([A-Z_]+)''')

def extract(raw,jid):
    ids=JID.findall(raw);preferences=PREF.findall(raw)
    if ids!=[jid] or len(preferences)!=1:return None
    value=ALIASES.get(preferences[0],preferences[0])
    return value if value in PREFS else None

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True,help='results/predictions/MODEL/ORDER')
    a=p.parse_args();source=a.folder/'results.csv';logs=a.folder/'attempts.jsonl'
    data=pd.read_csv(source,keep_default_na=False)
    if not data.pair_id.is_unique:raise ValueError('Duplicate pair_id')
    if not logs.exists():raise FileNotFoundError(logs)
    attempted={}
    for line in logs.read_text(encoding='utf-8').splitlines():
        if not line.strip():continue
        item=json.loads(line)
        # Claude batch failures include multiple IDs. Only one-item failures can
        # be safely recovered with the strict one-id parser.
        for pid in item.get('pair_ids',[]):attempted.setdefault(str(pid),[]).append(item)
    audit=[];unresolved=[];recovered=set()
    for idx,row in data.iterrows():
        if str(row['valid']).lower()=='true':continue
        records=attempted.get(str(row['pair_id']),[])
        extracted=[extract(str(t.get('raw_response','')),str(row['judgment_id'])) for t in records]
        if not records or None in extracted or len(set(extracted))!=1:
            unresolved.append(str(row['pair_id']))
            audit.append({'pair_id':row['pair_id'],'status':'unresolved','n_attempts':len(records)})
            continue
        pref=extracted[0];pos='A' if pref.startswith('A_') else 'B' if pref.startswith('B_') else 'EQUIVALENT'
        data.at[idx,'overall_preference']=pref
        data.at[idx,'predicted_fq_position']=pos
        data.at[idx,'correct']=pos==row['true_fq_position']
        data.at[idx,'valid']=True
        recovered.add(str(row['pair_id']))
        audit.append({'pair_id':row['pair_id'],'status':'recovered_preference_only',
                      'preference':pref,'n_attempts':len(records)})
    data['preference_recovered']=data.pair_id.astype(str).isin(recovered)
    data.to_csv(a.folder/'results_recovered.csv',index=False)
    (a.folder/'recovery_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'{len(recovered)} recovered; {len(unresolved)} unresolved; outputs are NOT overwritten')
    if unresolved:raise SystemExit(2)
if __name__=='__main__':main()
