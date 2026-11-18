"""Immutable QFrMark benchmark protocol, adapted from the original evaluation scripts."""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
import pandas as pd

PREFERENCES = ('A_MUCH_MORE_FQ','A_SLIGHTLY_MORE_FQ','EQUIVALENT','B_SLIGHTLY_MORE_FQ','B_MUCH_MORE_FQ')
DIMENSIONS = ('lexical','phraseological','syntactic_register','mixed','none')
REQUIRED = ('pair_id','feature_id','text_FI','text_FQ','marker_FI','marker_FQ','domaine')
SYSTEM_PROMPT = '''You are acting as an expert linguistic evaluator of International French (FI)
and Quebec French (FQ).

For every item, you receive exactly two anonymized sentences: Candidate A and
Candidate B. They express the same underlying content with a small linguistic
contrast.

PRIMARY TASK
Determine which candidate, if either, contains MORE LINGUISTIC CHARACTERISTICS
COMPATIBLE WITH QUEBEC FRENCH.

IMPORTANT
- Candidate order is randomized.
- You are not told which candidate is FQ or FI.
- Do not infer the answer from position.
- Do not assume Quebec French means slang or joual.
- Quebec French may be colloquial, standard, institutional, technical, or formal.
- Small but genuine lexical, phraseological, syntactic, or register differences
  are meaningful.
- Do not invent a difference when there is no clear evidence.

DIMENSIONS
1. lexical:
   vocabulary or terminology characteristic of Quebec French usage.
2. phraseological:
   multi-word expressions, idioms, collocations, or conventional turns of phrase.
3. syntactic_register:
   constructions or register choices compatible with Quebec French.
4. mixed:
   more than one dimension clearly contributes.
5. none:
   no meaningful FQ-oriented difference can be identified.

OVERALL PREFERENCE
Choose exactly one:
- A_MUCH_MORE_FQ
- A_SLIGHTLY_MORE_FQ
- EQUIVALENT
- B_SLIGHTLY_MORE_FQ
- B_MUCH_MORE_FQ

Use MUCH when the difference is strong and unambiguous.
Use SLIGHTLY when the difference is genuine but limited.
Use EQUIVALENT only when neither candidate shows a meaningful FQ advantage.

EVIDENCE SPANS
- fq_evidence_span: copy the shortest exact span from the candidate you consider
  more compatible with Quebec French that supports your decision.
- fi_evidence_span: when possible, copy the corresponding shortest exact span
  from the other candidate.
- If overall_preference is EQUIVALENT, return an empty string for both spans.
- Do not explain inside the evidence fields; copy text only.

RATIONALE
Give one concise linguistic explanation for the decision.

Your goal is not to guess a dataset label. Your goal is to identify genuine
linguistic evidence of Quebec French usage from the two sentences alone.'''

FIELDS = ('judgment_id','overall_preference','dominant_dimension','fq_evidence_span','fi_evidence_span','rationale')

def schema():
    fields = {k:{'type':'string'} for k in FIELDS}
    fields['overall_preference']['enum'] = list(PREFERENCES)
    fields['dominant_dimension']['enum'] = list(DIMENSIONS)
    return {'type':'object','properties':{'judgments':{'type':'array','items':{'type':'object',
            'properties':fields,'required':list(FIELDS),'additionalProperties':False}}},
            'required':['judgments'],'additionalProperties':False}

def user_prompt(jobs):
    items = [{'judgment_id':j['judgment_id'],'Candidate_A':j['candidate_a'],
              'Candidate_B':j['candidate_b']} for j in jobs]
    # Single-item wording and serialization reproduce the original common protocol.
    if len(items)==1:
        return ('Evaluate every pair independently according to the protocol.\n\n'
                'Do not assume that either candidate must be Quebec French. Use only linguistic '
                'evidence contained in the two sentences.\n\n'
                'Return a JSON object with a "judgments" array containing exactly one entry. '
                'Use keys judgment_id, overall_preference, dominant_dimension, '
                'fq_evidence_span, fi_evidence_span, rationale. No Markdown.\n\n'
                '<input_items>\n'+json.dumps(items,ensure_ascii=False)+'\n</input_items>')
    # Claude originally received batches of 10; text adapted from its original script.
    return ('Evaluate every pair independently according to the protocol.\n\n'
            'Do not assume that either candidate must be Quebec French. Use only linguistic\n'
            'evidence contained in the two sentences.\n\n'
            'Return exactly one judgment for every judgment_id.\n\n'
            '<input_items>\n'+json.dumps(items,ensure_ascii=False,indent=2)+'\n</input_items>')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_benchmark(path, strict=True):
    df=pd.read_csv(path,keep_default_na=False)
    absent=set(REQUIRED)-set(df.columns)
    if absent: raise ValueError(f'Missing columns: {sorted(absent)}')
    if df.pair_id.duplicated().any(): raise ValueError('Duplicate pair_id')
    if (df.text_FI.astype(str).str.strip()==df.text_FQ.astype(str).str.strip()).any():
        raise ValueError('Identical FI/FQ sentences')
    if strict and (len(df)!=1205 or df.feature_id.nunique()!=241 or not df.groupby('feature_id').size().eq(5).all()):
        raise ValueError('Expected 1205 pairs, 241 markers, five contexts each')
    return df

def make_jobs(df,seed=42):
    """Preserve original seed42 placement AND historical first-100 inversion IDs."""
    ix=list(range(len(df)));random.Random(seed).shuffle(ix)
    fq_a=set(ix[:len(ix)//2]); original=[]
    for i,row in df.reset_index(drop=True).iterrows():
        r=row.to_dict();a=i in fq_a
        r.update(judgment_id=f'J{i+1:05d}',phase='original',
          candidate_a=row.text_FQ if a else row.text_FI,
          candidate_b=row.text_FI if a else row.text_FQ,
          true_fq_position='A' if a else 'B',original_judgment_id='')
        original.append(r)
    # The original run selected 100 cases by random.Random(seed + 10000).
    first=random.Random(seed+10000).sample(original,min(100,len(original)))
    order=[j['pair_id'] for j in first]; chosen=set(order)
    order.extend(j['pair_id'] for j in original if j['pair_id'] not in chosen)
    by_id={j['pair_id']:j for j in original}; inverted=[]
    for i,pid in enumerate(order,1):
        j=dict(by_id[pid]); j.update(judgment_id=f'P{i:05d}',phase='inverted',
           candidate_a=j['candidate_b'], candidate_b=j['candidate_a'],
           true_fq_position='B' if j['true_fq_position']=='A' else 'A',
           original_judgment_id=j['judgment_id'])
        inverted.append(j)
    return original,inverted

def extract_json(raw):
    txt=raw.strip()
    if txt.startswith('```'):
        lines=txt.splitlines();lines=lines[1:]
        if lines and lines[-1].strip()=='```':lines=lines[:-1]
        txt='\n'.join(lines).strip()
    return json.loads(txt)

def validate_response(raw,jobs):
    obj=extract_json(raw)
    if not isinstance(obj,dict) or not isinstance(obj.get('judgments'),list):
        raise ValueError('Expected judgments JSON list')
    entries=obj['judgments']; required={j['judgment_id'] for j in jobs}
    ids=[e.get('judgment_id') for e in entries if isinstance(e,dict)]
    if len(entries)!=len(jobs) or set(ids)!=required or len(set(ids))!=len(ids):
        raise ValueError('Missing, duplicate or unexpected judgment ID')
    for e in entries:
        if set(e)!=set(FIELDS) or e['overall_preference'] not in PREFERENCES or e['dominant_dimension'] not in DIMENSIONS:
            raise ValueError('Invalid judgment structure or enum')
        if not all(isinstance(e[k],str) for k in FIELDS):raise ValueError('Non-string judgment field')
    return {e['judgment_id']:e for e in entries}

def prediction(pref):
    if pref.startswith('A_'):return 'A'
    if pref.startswith('B_'):return 'B'
    if pref=='EQUIVALENT':return 'EQUIVALENT'
    return 'INVALID'
