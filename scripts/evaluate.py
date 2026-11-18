#!/usr/bin/env python3
"""Run QFrMark originals, inversions, or both, using API or local HF models.

The same fixed dataset placements are used for all models. Claude retains its
historical batch-size 10; other models use one item/request. Invalid outputs
are recorded, never silently repaired or counted as valid judgments.
"""
from __future__ import annotations
import argparse, hashlib,json,os,sys,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from qfrmark.protocol import (SYSTEM_PROMPT,schema,user_prompt,validate_response,load_benchmark,
                              make_jobs,prediction,digest)

ALIASES={'llama31_8b','aya_expanse_8b','qwen3_8b','mistral_nemo_12b','gemma4_12b_it',
         'gpt54','gemini','claude'}
PROVIDERS={'gpt54':'openai','gemini':'gemini','claude':'anthropic',
           'llama31_8b':'local','aya_expanse_8b':'local','qwen3_8b':'local',
           'mistral_nemo_12b':'local','gemma4_12b_it':'gemma'}

def append(path,rec):
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')

def load_records(path):
    data={}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip(): continue
            obj=json.loads(line);key=obj['pair_id']
            if key in data:raise ValueError(f'Duplicate pair_id in {path}: {key}')
            data[key]=obj
    return data

def guarded_config(out,a,benchmark_sha):
    cfg={'model_key':a.model,'provider':PROVIDERS[a.model], 'checkpoint':a.checkpoint,
         'benchmark_sha256':benchmark_sha,'seed':a.seed,'batch_size':a.batch_size,
         'max_new_tokens':a.max_new_tokens,'max_retries':a.max_retries,
         'protocol':'qfrmark_original_public_v1', 'reasoning_effort':'none' if a.model=='gpt54' else None,
         'gemini_thinking_level':a.thinking_level if a.model=='gemini' else None}
    path=out/'run_config.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=cfg:
        raise ValueError(f'Run configuration has changed; use a NEW output directory: {out}')
    path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')

def make_caller(a):
    key=a.model
    if key=='gpt54':
        from openai import OpenAI
        if not os.getenv('OPENAI_API_KEY'):raise ValueError('OPENAI_API_KEY not set')
        client=OpenAI(timeout=120,max_retries=2)
        def call(jobs):
            r=client.responses.create(model=a.checkpoint,reasoning={'effort':'none'},
              instructions=SYSTEM_PROMPT,input=user_prompt(jobs),
              text={'format':{'type':'json_schema','name':'fq_fi_judgment','strict':True,'schema':schema()}},
              max_output_tokens=a.max_new_tokens)
            if not r.output_text:raise RuntimeError('OpenAI output_text empty')
            return r.output_text
        return call
    if key=='gemini':
        from google import genai
        from google.genai import types
        api_key=os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:raise ValueError('GEMINI_API_KEY not set')
        client=genai.Client(api_key=api_key)
        def call(jobs):
            r=client.models.generate_content(model=a.checkpoint,contents=user_prompt(jobs),
             config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT,
              response_mime_type='application/json',response_json_schema=schema(),
              thinking_config=types.ThinkingConfig(thinking_level=a.thinking_level),
              max_output_tokens=a.max_new_tokens))
            if not r.text:raise RuntimeError('Gemini response empty')
            return r.text
        return call
    if key=='claude':
        from anthropic import Anthropic
        if not os.getenv('ANTHROPIC_API_KEY'):raise ValueError('ANTHROPIC_API_KEY not set')
        client=Anthropic(api_key=os.environ['ANTHROPIC_API_KEY'])
        def call(jobs):
            r=client.messages.create(model=a.checkpoint,max_tokens=a.claude_max_tokens,
             system=SYSTEM_PROMPT,messages=[{'role':'user','content':user_prompt(jobs)}],
             output_config={'format':{'type':'json_schema','schema':schema()}})
            return '\n'.join(part.text for part in r.content if getattr(part,'type',None)=='text').strip()
        return call
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required for local inference')
    if key=='gemma4_12b_it':
        from transformers import AutoProcessor,AutoModelForMultimodalLM
        processor=AutoProcessor.from_pretrained(a.checkpoint,trust_remote_code=False,local_files_only=a.offline)
        model=AutoModelForMultimodalLM.from_pretrained(a.checkpoint,trust_remote_code=False,
            local_files_only=a.offline,dtype='auto',device_map='auto').eval()
        def call(jobs):
            messages=[{'role':'system','content':SYSTEM_PROMPT},
                      {'role':'user','content':user_prompt(jobs)}]
            inputs=processor.apply_chat_template(messages,tokenize=True,return_dict=True,
              return_tensors='pt',add_generation_prompt=True).to(model.device)
            length=inputs['input_ids'].shape[-1]
            with torch.inference_mode():
                output=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,
                   pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id)
            return processor.decode(output[0][length:],skip_special_tokens=True)
        return call
    from transformers import AutoTokenizer,AutoModelForCausalLM
    tok_kwargs={'trust_remote_code':False,'local_files_only':a.offline}
    if key=='mistral_nemo_12b':tok_kwargs['fix_mistral_regex']=True
    tokenizer=AutoTokenizer.from_pretrained(a.checkpoint,**tok_kwargs)
    if tokenizer.pad_token_id is None:tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(a.checkpoint,local_files_only=a.offline,
      dtype=torch.bfloat16,device_map='auto',trust_remote_code=False).eval()
    def call(jobs):
        messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':user_prompt(jobs)}]
        extra={'enable_thinking':False} if key=='qwen3_8b' else {}
        rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,**extra)
        inputs=tokenizer(rendered,return_tensors='pt').to(model.device)
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=a.max_new_tokens,do_sample=False,
               pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(output[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
    return call

def run_order(a,jobs,call):
    folder=a.out/a.model/jobs[0]['phase'];folder.mkdir(parents=True,exist_ok=True)
    path=folder/'records.jsonl';attempt_path=folder/'attempts.jsonl'
    all_jobs=jobs[:a.limit] if a.limit else jobs
    existing=load_records(path)
    # Retry invalid outputs ONLY if asked; preserve an audit instead of overwriting.
    if a.retry_invalid and any(not r['valid'] for r in existing.values()):
        raise ValueError('Invalid records already exist: keep original immutable; use a new --out for a clean rerun or repair separately.')
    pending=[j for j in all_jobs if j['pair_id'] not in existing]
    print(f'{a.model} {jobs[0]["phase"]}: {len(existing)} saved; {len(pending)} pending',flush=True)
    for offset in range(0,len(pending),a.batch_size):
        batch=pending[offset:offset+a.batch_size];raw='';parsed=None;error=''
        for attempt in range(1,a.max_retries+1):
            try:
                raw=call(batch);parsed=validate_response(raw,batch);break
            except KeyboardInterrupt: raise
            except Exception as ex:
                error=f'{type(ex).__name__}: {ex}'
                append(attempt_path,{'pair_ids':[j['pair_id'] for j in batch],
                     'attempt':attempt,'error':error,'raw_response':raw})
                if attempt<a.max_retries:time.sleep(min(30,2**attempt))
        for j in batch:
            entry=parsed[j['judgment_id']] if parsed else {k:'' for k in
                ('judgment_id','overall_preference','dominant_dimension','fq_evidence_span','fi_evidence_span','rationale')}
            entry['judgment_id']=j['judgment_id']
            pref=entry['overall_preference'];pos=prediction(pref)
            rec={**entry,'pair_id':j['pair_id'],'feature_id':j['feature_id'],'domaine':j['domaine'],
                 'candidate_a':j['candidate_a'],'candidate_b':j['candidate_b'],
                 'true_fq_position':j['true_fq_position'],'predicted_fq_position':pos,
                 'correct':pos==j['true_fq_position'],'valid':pos!='INVALID',
                 'original_judgment_id':j['original_judgment_id'],'model':a.model,
                 'phase':j['phase'],'error':error if parsed is None else ''}
            append(path,rec)
        if a.sleep_seconds:time.sleep(a.sleep_seconds)
        if (offset//a.batch_size+1)%20==0:print(f'  {offset+len(batch)}/{len(pending)} new',flush=True)
    rows=load_records(path)
    df=pd.DataFrame([rows[j['pair_id']] for j in all_jobs if j['pair_id'] in rows])
    df.to_csv(folder/'results.csv',index=False)
    print(f'{folder}: {len(df)} rows, {int(df.valid.sum())} valid',flush=True)
    if len(df)!=len(all_jobs) or not df.valid.astype(bool).all():
        raise RuntimeError(f'Incomplete/invalid results in {folder}; see attempts.jsonl')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,default=Path('data/qfrmark.csv'))
    p.add_argument('--out',type=Path,default=Path('results/predictions'))
    p.add_argument('--model',choices=sorted(ALIASES),required=True)
    p.add_argument('--checkpoint',required=True,help='EXACT API ID or HF model ID / local checkpoint path')
    p.add_argument('--order',choices=['original','inverted','both'],default='both')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-new-tokens',type=int,default=768)
    p.add_argument('--claude-max-tokens',type=int,default=8000)
    p.add_argument('--thinking-level',choices=['low','medium','high'],default='low')
    p.add_argument('--batch-size',type=int,default=None)
    p.add_argument('--max-retries',type=int,default=5)
    p.add_argument('--sleep-seconds',type=float,default=0)
    p.add_argument('--limit',type=int,help='First N jobs for a test; use a separate --out from full')
    p.add_argument('--offline',action='store_true',help='Use locally cached checkpoint only')
    p.add_argument('--retry-invalid',action='store_true',help='Do not overwrite; errors need audited recovery')
    a=p.parse_args()
    if a.batch_size is None:a.batch_size=10 if a.model=='claude' else 1
    if a.model!='claude' and a.batch_size!=1:p.error('Only Claude supports batched requests')
    if a.seed!=42:print('WARNING: changes candidate placement from the publication',file=sys.stderr)
    df=load_benchmark(a.benchmark,strict=a.limit is None)
    main,inv=make_jobs(df,a.seed)
    folder=a.out/a.model;folder.mkdir(parents=True,exist_ok=True)
    guarded_config(folder,a,digest(a.benchmark))
    call=make_caller(a)
    if a.order in ('original','both'):run_order(a,main,call)
    if a.order in ('inverted','both'):run_order(a,inv,call)
if __name__=='__main__':main()
