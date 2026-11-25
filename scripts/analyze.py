#!/usr/bin/env python3
"""Validate paired model predictions, compute all full-inversion analyses and plots.

No inference is performed. Invalid/incomplete inputs fail closed. Statistical
unit for bootstrap and between-model sign-flip tests: linguistic marker (241).
"""
from __future__ import annotations
import argparse,itertools,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd
from scipy.stats import binomtest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qfrblima.protocol import PREFERENCES,load_benchmark,prediction,digest
from plot_publication_figures import generate_figures

MODELS=['gemini','claude','gpt54','gemma4_12b_it','mistral_nemo_12b','qwen3_8b','aya_expanse_8b','llama31_8b']
LABELS={'gemini':'Gemini 3.8 Flash','claude':'Claude Opus 4.8','gpt54':'GPT-5.4','gemma4_12b_it':'Gemma 4 12B IT',
        'mistral_nemo_12b':'Mistral NeMo 12B','qwen3_8b':'Qwen3 8B','aya_expanse_8b':'Aya Expanse 8B','llama31_8b':'Llama 3.1 8B'}

def holm(ps):
    p=np.asarray(ps,dtype=float);idx=np.argsort(p);out=np.zeros(len(p));prev=0.
    for rank,j in enumerate(idx):
        prev=max(prev,min(1.,(len(p)-rank)*p[j]));out[j]=prev
    return out

def exact_mcnemar(a_only,b_only):
    n=a_only+b_only
    return float(binomtest(min(a_only,b_only),n,.5).pvalue) if n else 1.

def read_valid(folder,expected,benchmark,order):
    path=(folder/order/'results_recovered.csv' if (folder/order/'results_recovered.csv').is_file()
          else folder/order/'results.csv')
    if not path.exists():raise FileNotFoundError(path)
    df=pd.read_csv(path,keep_default_na=False)
    required={'pair_id','feature_id','judgment_id','candidate_a','candidate_b','overall_preference',
              'true_fq_position','original_judgment_id','valid'}
    if required-set(df.columns):raise ValueError(f'{path}: missing {required-set(df.columns)}')
    if len(df)!=len(benchmark) or not df.pair_id.is_unique or set(df.pair_id)!=set(benchmark.pair_id):
        raise ValueError(f'{path}: incomplete/duplicate/mismatched pair_ids')
    df=df.set_index('pair_id').loc[benchmark.pair_id].reset_index()
    if not df.overall_preference.isin(PREFERENCES).all() or not df.valid.astype(str).str.lower().eq('true').all():
        raise ValueError(f'{path}: invalid model outputs')
    if not df.feature_id.eq(benchmark.feature_id).all():raise ValueError('feature_id mismatch')
    for side,col in [('a','candidate_a'),('b','candidate_b')]:
        expected_text=np.where(df.true_fq_position.eq(side.upper()),benchmark.text_FQ,benchmark.text_FI)
        if not np.array_equal(df[col].astype(str).to_numpy(),expected_text):
            raise ValueError(f'{path}: unexpected candidate texts')
    df['pred']=df.overall_preference.map(prediction)
    df['correct_calc']=df.pred.eq(df.true_fq_position)
    if 'correct' in df and not df.correct.astype(str).str.lower().map({'true':True,'false':False}).eq(df.correct_calc).all():
        raise ValueError(f'{path}: incorrect correctness column')
    return df

def pairs_for_model(root,key,benchmark):
    o=read_valid(root/key,None,benchmark,'original')
    i=read_valid(root/key,None,benchmark,'inverted')
    if not i.true_fq_position.eq(o.true_fq_position.map({'A':'B','B':'A'})).all():
        raise ValueError(f'{key}: gold labels not inverted')
    if not i.candidate_a.eq(o.candidate_b).all() or not i.candidate_b.eq(o.candidate_a).all():
        raise ValueError(f'{key}: strings not swapped')
    if not i.original_judgment_id.eq(o.judgment_id).all():raise ValueError('original_judgment_id mismatch')
    swap=o.pred.map({'A':'B','B':'A','EQUIVALENT':'EQUIVALENT'})
    x=pd.DataFrame({'model':key,'pair_id':o.pair_id,'feature_id':o.feature_id,
        'domaine':benchmark.domaine,'marker_FI':benchmark.marker_FI,'marker_FQ':benchmark.marker_FQ,
        'original_correct':o.correct_calc,'inverted_correct':i.correct_calc,
        'original_prediction':o.pred,'inverted_prediction':i.pred,
        'equivalent_original':o.pred.eq('EQUIVALENT'),
        'position_consistent':i.pred.eq(swap),
        'correct_both':o.correct_calc & i.correct_calc,
        'correct_one':o.correct_calc ^ i.correct_calc,
        'incorrect_both':~o.correct_calc & ~i.correct_calc,
        'fq_in_a_correct':o.correct_calc.where(o.true_fq_position.eq('A'),i.correct_calc),
        'fq_in_b_correct':i.correct_calc.where(o.true_fq_position.eq('A'),o.correct_calc),
        'original_fq_position':o.true_fq_position})
    return x

def boot_ci(matrix,B,rng):
    # matrix: F markers x K metrics; all markers have exactly five contexts.
    F,K=matrix.shape;sample=np.empty((B,K))
    for start in range(0,B,200):
        end=min(B,start+200);ids=rng.integers(0,F,size=(end-start,F))
        sample[start:end]=matrix[ids].mean(axis=1)
    return np.quantile(sample,[.025,.975],axis=0)

DOMAIN_LABELS = {
    'sport': 'Sports',
    'loisir': 'Leisure',
    'alimentation': 'Food',
    'zoologie': 'Zoology',
    'médecine': 'Medicine',
    'industrie de la construction': 'Construction',
    'télécommunication': 'Telecommunications',
    'botanique': 'Botany',
    'administration publique': 'Public administration',
    'comptabilité': 'Accounting',
    'route': 'Road transport',
    'éducation': 'Education',
    'entreprise': 'Business',
    'commerce': 'Commerce',
    'finance': 'Finance',
    'bureau': 'Office work',
    'droit': 'Law',
    'industrie automobile': 'Automotive industry',
    'bijouterie': 'Jewelry',
    'emballage et conditionnement': 'Packaging',
    'gestion': 'Management',
    'économie': 'Economics',
}

def plot_both(overall,out):
    s=overall.sort_values('correct_both',ascending=True)
    names=s.model.map(LABELS)
    y=np.arange(len(s))
    fig,ax=plt.subplots(figsize=(10.5,5.2))
    parts=[
        (s.correct_both.to_numpy()*100,'Correct in both orders'),
        (s.correct_one.to_numpy()*100,'Correct in one order'),
        (s.incorrect_both.to_numpy()*100,'Incorrect in both orders'),
    ]
    left=np.zeros(len(s))
    for values,label in parts:
        bars=ax.barh(y,values,left=left,label=label,height=.7)
        for j,(bar,value) in enumerate(zip(bars,values)):
            if value>=5:
                ax.text(left[j]+value/2,bar.get_y()+bar.get_height()/2,
                        f'{value:.1f}%',ha='center',va='center',
                        fontsize=8,fontweight='400',
                        color='white' if np.mean(bar.get_facecolor()[:3])<.5 else 'black')
        left+=values
    ax.set(yticks=y,yticklabels=names,xlim=(0,100),xlabel='Share of sentence pairs (%)')
    ax.legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.13),ncol=3,frameon=False)
    fig.tight_layout()
    fig.savefig(out/'order_robustness.png',dpi=250,bbox_inches='tight')
    plt.close(fig)

def plot_domains(domain,out):
    size=domain[domain.model=='gemini']
    chosen=(size.query('n_markers >= 5')
            .sort_values('n_pairs',ascending=False).head(15).domain.tolist())
    if not chosen:return
    mat=(domain.pivot(index='domain',columns='model',values='robust')
         .reindex(index=chosen,columns=MODELS)*100)
    english=[DOMAIN_LABELS.get(name,name) for name in mat.index]
    fig,ax=plt.subplots(figsize=(11.4,max(5.6,.43*len(chosen)+1.5)))
    im=ax.imshow(mat.to_numpy(),vmin=0,vmax=100,aspect='auto',cmap='viridis')
    ax.set_yticks(range(len(mat)),english)
    ax.set_xticks(range(len(MODELS)),[LABELS[m] for m in MODELS],rotation=38,ha='right')
    for j in range(len(mat)):
        for k in range(len(MODELS)):
            value=mat.iloc[j,k]
            if pd.notna(value):
                ax.text(k,j,f'{value:.0f}',ha='center',va='center',fontsize=8,
                        color='white' if value<55 else 'black')
    fig.colorbar(im,ax=ax,label='Correct in both orders (%)',shrink=.9)
    ax.set_title('Order-robust recognition by terminological domain')
    fig.tight_layout()
    fig.savefig(out/'domain_heatmap.png',dpi=250,bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',type=Path,default=Path('data/qfrblima.csv'))
    p.add_argument('--predictions',type=Path,default=Path('results/predictions'))
    p.add_argument('--out',type=Path,default=Path('results/analysis'))
    p.add_argument('--models',nargs='+',default=MODELS)
    p.add_argument('--bootstrap',type=int,default=10000)
    p.add_argument('--permutations',type=int,default=20000)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.bootstrap<100 or a.permutations<100:raise ValueError('Use at least 100 resamples')
    b=load_benchmark(a.benchmark,strict=True).reset_index(drop=True)
    frames=[pairs_for_model(a.predictions,m,b) for m in a.models]
    all_pairs=pd.concat(frames,ignore_index=True)
    all_pairs.to_csv(a.out/'paired_observations.csv',index=False)
    rng=np.random.default_rng(a.seed)
    summaries=[];tests=[];ci=[];marker=[];domains=[]
    for m,frame in zip(a.models,frames):
        first=frame.original_correct.to_numpy(bool);inv=frame.inverted_correct.to_numpy(bool)
        ca=frame.fq_in_a_correct.to_numpy(bool);cb=frame.fq_in_b_correct.to_numpy(bool)
        orig_only=int(np.sum(first & ~inv));inv_only=int(np.sum(~first & inv))
        a_only=int(np.sum(ca & ~cb));b_only=int(np.sum(~ca & cb))
        summaries.append({'model':m,'n':len(frame),'original':first.mean(),'inverted':inv.mean(),
            'correct_both':frame.correct_both.mean(),'correct_one':frame.correct_one.mean(),
            'incorrect_both':frame.incorrect_both.mean(),'position_consistency':frame.position_consistent.mean(),
            'fq_in_a':ca.mean(),'fq_in_b':cb.mean(),'equivalent_original':frame.equivalent_original.mean()})
        tests.append({'model':m,'main_only':orig_only,'inverted_only':inv_only,
            'p_main_vs_inverted':exact_mcnemar(orig_only,inv_only),
            'fq_a_only':a_only,'fq_b_only':b_only,
            'p_fq_a_vs_b':exact_mcnemar(a_only,b_only)})
        gm=frame.groupby('feature_id',sort=True)
        mat=gm[['original_correct','inverted_correct','correct_both','position_consistent']].mean()
        ci_limits=boot_ci(mat.to_numpy(),a.bootstrap,rng)
        for k,col in enumerate(mat.columns):
            ci.append({'model':m,'metric':col,'estimate':mat[col].mean(),
                       'low95':ci_limits[0,k],'high95':ci_limits[1,k]})
        for marker_id,g in gm:
            cnt=int(g.correct_both.sum());
            marker.append({'model':m,'feature_id':marker_id,'original_correct_n':int(g.original_correct.sum()),
              'robust_contexts_n':cnt,'robust_class':'full' if cnt==5 else 'none' if cnt==0 else 'partial',
              'incorrect_both_n':int(g.incorrect_both.sum()),'n_contexts':5})
        # Preserve the original script's mutually exclusive principal domain rule.
        domains_=(frame.domaine.fillna('').astype(str).str.split(';',regex=False).str[0]
                    .str.split('>',regex=False).str[0].str.strip().replace('', 'unspecified'))
        temp=frame.assign(domain=domains_)
        for domain,g in temp.groupby('domain'):
            domains.append({'model':m,'domain':domain,'n_pairs':len(g),'n_markers':g.feature_id.nunique(),
                      'robust':g.correct_both.mean(),'original':g.original_correct.mean(),
                      'inverted':g.inverted_correct.mean()})
    overall=pd.DataFrame(summaries);t=pd.DataFrame(tests);t['holm_main_vs_inverted']=holm(t.p_main_vs_inverted)
    t['holm_fq_a_vs_b']=holm(t.p_fq_a_vs_b)
    original_models=a.models
    # Paired marker-cluster sign-flip permutation tests on robust accuracy.
    cluster={m:frames[i].groupby('feature_id').correct_both.mean().sort_index()
             for i,m in enumerate(original_models)}
    for v in cluster.values():
        if len(v)!=241 or not v.index.equals(next(iter(cluster.values())).index):
            raise ValueError('Cluster mismatch across models')
    comparison=[]
    for m1,m2 in itertools.combinations(original_models,2):
        diff=(cluster[m1]-cluster[m2]).to_numpy(dtype=float);delta=float(diff.mean())
        extreme=0
        for start in range(0,a.permutations,300):
            size=min(300,a.permutations-start)
            signs=rng.choice([-1.,1.],size=(size,len(diff)))
            extreme+=int(np.count_nonzero(np.abs(signs@diff/len(diff))>=abs(delta)-1e-12))
        comparison.append({'model_1':m1,'model_2':m2,'delta_robust':delta,
                           'p_cluster_permutation':(extreme+1)/(a.permutations+1)})
    comp=pd.DataFrame(comparison)
    if len(comp):comp['holm_p']=holm(comp.p_cluster_permutation)
    features=all_pairs.groupby('feature_id').agg(n=('pair_id','size'),
      incorrect_both_n=('incorrect_both','sum'),correct_both_n=('correct_both','sum'),
      n_models_robust_any=('model',lambda _:0))
    features=features.drop(columns=['n_models_robust_any'])
    features['incorrect_both_rate']=features.incorrect_both_n/features.n
    hard=features.sort_values('incorrect_both_rate',ascending=False).reset_index()
    pairs_agg=all_pairs.groupby('pair_id').agg(n_models_robust=('correct_both','sum'),
        n_models_incorrect_both=('incorrect_both','sum')).reset_index()
    for filename,df in [('overall.csv',overall),('paired_mcnemar_holm.csv',t),
      ('cluster_bootstrap_ci.csv',pd.DataFrame(ci)),('pairwise_cluster_permutation_holm.csv',comp),
      ('marker_recognition.csv',pd.DataFrame(marker)),('domain_performance.csv',pd.DataFrame(domains)),
      ('difficult_markers.csv',hard),('shared_pair_errors.csv',pairs_agg)]:
        df.to_csv(a.out/filename,index=False)
    plot_both(overall,a.out)  # Keep the legacy two-order stacked bar figure.
    # Paper figures: two-panel accuracy, bootstrap CIs, and 12-category heatmap.
    # IMPORTANT: do not call plot_domains() here: it would overwrite the new heatmap.
    generate_figures(a.out, Path('data/domain_to_broad_category.csv'))
    manifest={'seed':a.seed,'benchmark_sha256':digest(a.benchmark),'models':a.models,
      'n_pairs':len(b),'n_markers':b.feature_id.nunique(),
      'n_models':len(a.models),'n_pairs_robust_for_none':int((pairs_agg.n_models_robust==0).sum()),
      'n_pairs_robust_for_all':int((pairs_agg.n_models_robust==len(a.models)).sum()),
      'statistical_unit':'feature_id (241 marker clusters)',
      'domain_assignment':'first semicolon-separated domain; before >',
      'interpretation':'McNemar original vs inverted and FQ A vs B are distinct tests; domain analysis descriptive'}
    (a.out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(overall.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('Saved',a.out)
if __name__=='__main__':main()
