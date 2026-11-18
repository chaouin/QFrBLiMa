import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from qfrmark.protocol import make_jobs,schema,validate_response, user_prompt

def test_balance_and_inversion():
    df=pd.DataFrame({'pair_id':[str(i) for i in range(1205)],
                     'feature_id':[str(i//5) for i in range(1205)],
                     'text_FI':['fi'+str(i) for i in range(1205)],
                     'text_FQ':['fq'+str(i) for i in range(1205)]})
    a,b=make_jobs(df)
    assert len(a)==len(b)==1205
    assert sum(j['true_fq_position']=='A' for j in a)==602
    assert len(set(j['pair_id'] for j in b))==1205
    mp={j['pair_id']:j for j in a}
    for j in b:
        x=mp[j['pair_id']]
        assert x['candidate_a']==j['candidate_b']
        assert x['candidate_b']==j['candidate_a']
        assert j['original_judgment_id']==x['judgment_id']

def test_strict_json():
    job={'judgment_id':'J00001','candidate_a':'Bonjour','candidate_b':'Allô'}
    import json
    data={'judgments':[{'judgment_id':'J00001','overall_preference':'B_MUCH_MORE_FQ',
                       'dominant_dimension':'lexical','fq_evidence_span':'Allô',
                       'fi_evidence_span':'Bonjour','rationale':'Example'}]}
    assert validate_response(json.dumps(data),[job])['J00001']['overall_preference']=='B_MUCH_MORE_FQ'
    data['judgments'][0]['overall_preference']='INVALID'
    import pytest
    with pytest.raises(ValueError):validate_response(json.dumps(data),[job])
