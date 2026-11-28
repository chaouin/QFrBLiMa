#!/usr/bin/env python3
"""Five manuscript figures from released predictions and analysis CSVs. No inference.
Run from repo root: python scripts/plot_figures.py
Run analyze_preference_strength.py first if preference_strength.csv is absent.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
L=['gemini','claude','gpt54','gemma4_12b_it','mistral_nemo_12b','qwen3_8b','aya_expanse_8b','llama31_8b']
E=['camembert','camembertv2','moderncamembert','roberta_base','bert_base_cased']
N=['Gemini 3.8 Flash','Claude Opus 4.8','GPT-5.4','Gemma 4 12B IT','Mistral NeMo 12B Instruct','Qwen3 8B','Aya Expanse 8B','Llama 3.1 8B Instruct','CamemBERT','CamemBERTv2','ModernCamemBERT','RoBERTa-base','BERT-base-cased']
SHORT=['Gemini','Claude','GPT-5.4','Gemma 4','Mistral NeMo','Qwen3','Aya Expanse','Llama 3.1','CamemBERT','CamemBERTv2','ModernCamemBERT','RoBERTa','BERT']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
Y=np.r_[np.arange(8),np.arange(9,14)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--out',type=Path,default=Path('results/analysis'));p.add_argument('--preference',type=Path);p.add_argument('--analysis',type=Path,help='Input analysis CSV directory (default: ROOT/results/analysis)');a=p.parse_args();r=a.root;src=a.analysis or r/'results/analysis';out=a.out;out.mkdir(parents=True,exist_ok=True)
 def save(fig,name):
  for ext in ['pdf','png']:fig.savefig(out/f'{name}.{ext}',bbox_inches='tight',pad_inches=.035)
  plt.close(fig)
 def style(ax,y,n,xmax=100,xlabel='Accuracy (%)'):
  ax.set(yticks=y,yticklabels=n,xlim=(0,xmax),xlabel=xlabel);ax.invert_yaxis();ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
 def stacked(ax,arr,y,n,labels,colors,xmax=100,xlabel='Share (%)',fmt='.1f',minval=7):
  left=np.zeros(len(y))
  for j,(lab,col) in enumerate(zip(labels,colors)):
   v=arr[:,j];ax.barh(y,v,left=left,height=.72,color=col,label=lab)
   for i,w in enumerate(v):
    if w>=minval:
     rgb=matplotlib.colors.to_rgb(col); lum=sum(x*z for x,z in zip(rgb,[.299,.587,.114]));ax.text(left[i]+w/2,y[i],format(w,fmt),ha='center',va='center',fontsize=8.5,color='white' if lum<.48 else '#161616')
   left+=v
  style(ax,y,n,xmax,xlabel)
 overall=pd.read_csv(src/'overall.csv').set_index('model').loc[L];pairs=pd.read_csv(src/'paired_observations.csv');ci=pd.read_csv(src/'cluster_bootstrap_ci.csv').set_index(['model','metric']);b=pd.read_csv(r/'data/qfrblima.csv');mapping=pd.read_csv(r/'data/domain_to_broad_category.csv').set_index('primary_domain').broad_category
 def cats(s):return s.str.split(';').str[0].str.split('>').str[0].str.strip().map(mapping)
 b['category']=cats(b.domaine);assert b.category.notna().all() and b.pair_id.is_unique and len(b)==1205
 categories=sorted(b.category.unique());counts=b.groupby('category').feature_id.nunique().reindex(categories)
 enc={};met={};train={}
 for m in E:
  d=r/'results/encoders/evaluation'/m;enc[m]=pd.read_csv(d/'paired_predictions.csv').merge(b[['pair_id','category']],on='pair_id',validate='one_to_one');met[m]=json.loads((d/'overall_metrics.json').read_text());train[m]=json.loads((r/'results/encoders/training'/m/'test_metrics.json').read_text());assert len(enc[m])==1205
 pref=pd.read_csv(a.preference or src/'preference_strength.csv').set_index('model').loc[N];pc=['strong_correct','slight_correct','no_preference','slight_incorrect','strong_incorrect'];pv=pref[[c+'_pct' for c in pc]].to_numpy();assert np.allclose(pv.sum(1),100) and np.array_equal(pref.n.to_numpy(),[2410]*8+[1205]*5)
 # Figure 1: overall accuracy.
 fig,ax=plt.subplots(figsize=(7.2,4.15))
 for i,m in enumerate(L):
  d=pairs[pairs.model==m];g=d.groupby('feature_id')[['original_correct','inverted_correct']].mean().mean(axis=1).to_numpy();assert len(g)==241
  rng=np.random.default_rng(42);vals=g[rng.integers(0,len(g),size=(20000,len(g)))].mean(axis=1);mean=g.mean()*100;lo,hi=np.percentile(vals,[2.5,97.5])*100
  robust=ci.loc[(m,'correct_both')]*100
  for v,l,h,yy,col,marker,dy in [(mean,lo,hi,Y[i]-.15,'#4477aa','o',6),(robust.estimate,robust.low95,robust.high95,Y[i]+.16,'#2299bb','s',-6)]:
   ax.errorbar(v,yy,xerr=[[v-l],[h-v]],fmt=marker,color=col,capsize=2,ms=4)
   right=marker=='o' or l<10
   ax.annotate(f'{v:.2f}',(h if right else l,yy),xytext=(3 if right else -3,0),textcoords='offset points',ha='left' if right else 'right',va='center',fontsize=8,color=col)
 for j,m in enumerate(E):
  v=met[m]['pairwise_FQ_higher']*100;l,h=np.array(met[m]['pairwise_marker_cluster_95ci'])*100;ax.errorbar(v,Y[8+j],xerr=[[v-l],[h-v]],fmt='D',color='#228833',capsize=2,ms=4);ax.annotate(f'{v:.2f}',(v,Y[8+j]),xytext=(0,6),textcoords='offset points',ha='center',fontsize=8,color='#228833')
 style(ax,Y,SHORT,112);ax.set_xticks(range(0,101,20));ax.axhline(8,color='.75',lw=.8)
 ax.legend(handles=[Line2D([],[],marker=k,color=c,label=t,ls='none') for k,c,t in [('o','#4477aa','LLM mean single-order'),('s','#2299bb','LLM order-robust'),('D','#228833','Encoder strict pairwise')]],loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,frameon=False,fontsize=9)
 save(fig,'figure1_overall_accuracy')
 # Figure 2: preference strength, same model order.
 fig,bx=plt.subplots(figsize=(7.2,3.55))
 stacked(bx,pv,Y,SHORT,['Strong correct','Slight correct','No preference','Slight incorrect','Strong incorrect'],['#26734d','#a6d4ae','#c4c4c4','#efaa91','#ae3434'],xlabel='Judgments or pair rankings (%)',minval=7);bx.axhline(8,color='.75',lw=.8);bx.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,frameon=False,fontsize=9);save(fig,'figure2_preference_strength')
 # Figure 3: paired-order outcomes and marginal accuracies; no transfer data.
 fig,ax=plt.subplots(figsize=(7.2,3.3));arr=overall[['correct_both','correct_one','incorrect_both']].to_numpy()*100;arr=np.vstack([arr,[25,50,25],[16,48,36],[0,100,0]]);y=np.r_[np.arange(8),np.arange(9,12)];stacked(ax,arr,y,SHORT[:8]+['Random A/B','Random 5-label','Always A'],['Both correct','Exactly one correct','Neither correct'],['#4477aa','#ddbb66','#bbbbbb'],xlabel='Sentence pairs (%)',minval=5);ax.axhline(8,color='.75',lw=.8);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.21),ncol=3,frameon=False);marg=np.vstack([overall[['original','inverted']].to_numpy()*100,[50,50],[40,40],[100*602/1205,100*603/1205]]);ax.text(1.03,1.035,'Original',transform=ax.transAxes,fontsize=9,ha='left');ax.text(1.19,1.035,'Inverted',transform=ax.transAxes,fontsize=9,ha='left');
 for i,yy in enumerate(y):
  for x,j in [(1.03,0),(1.19,1)]:ax.text(x,yy,f'{marg[i,j]:.2f}',transform=ax.get_yaxis_transform(),va='center',fontsize=9)
 save(fig,'figure3_candidate_order')
 # Classification transfer: export exact table values instead of a figure.
 transfer=pd.DataFrame({'model':N[8:],'internal_macro_f1_pct':[train[m]['f1_macro']*100 for m in E],'external_macro_f1_pct':[met[m]['f1_macro']*100 for m in E],'external_accuracy_pct':[met[m]['accuracy']*100 for m in E]})
 transfer.to_csv(out/'classification_transfer.csv',index=False)
 # Figure 4: marker-level context consistency.
 frames=[pairs[pairs.model==m].groupby('feature_id').correct_both.sum() for m in L]+[enc[m].groupby('feature_id').pairwise_success.sum() for m in E];arr=np.array([[sum(g==5),sum((g>0)&(g<5)),sum(g==0)] for g in frames]);assert np.all(arr.sum(1)==241)
 fig,ax=plt.subplots(figsize=(7.2,3.6));stacked(ax,arr,Y,SHORT,['Full (5/5)','Partial (1–4/5)','None (0/5)'],['#4477aa','#ddbb66','#bbbbbb'],241,'Number of linguistic markers',fmt='d',minval=999);ax.set_xticks([0,50,100,150,200,241]);ax.axhline(8,color='.75',lw=.8);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.20),ncol=3,frameon=False)
 for j,lab in enumerate(['Full','Partial','None']):
  x=1.035+j*.095;ax.text(x,1.025,lab,transform=ax.transAxes,ha='center',fontsize=9)
  for i,yy in enumerate(Y):ax.text(x,yy,str(arr[i,j]),transform=ax.get_yaxis_transform(),ha='center',va='center',fontsize=9)
 save(fig,'figure4_contextual_consistency')
 # Figure 5: model/category heatmap with group means, computed before rounding.
 pairs=pairs.merge(b[['pair_id','category']],on='pair_id',validate='many_to_one');pairs['single']=(pairs.original_correct.astype(int)+pairs.inverted_correct.astype(int))/2
 mats=[np.array([pairs[pairs.model==m].groupby('category')[k].mean().reindex(categories) for m in L]).T*100 for k in ['single','correct_both']];mats.append(np.array([enc[m].groupby('category').pairwise_success.mean().reindex(categories) for m in E]).T*100)
 assert all(np.isfinite(mat).all() for mat in mats)
 thematic_rows=[]
 for metric,mat,models in zip(['llm_mean_single_order','llm_order_robust','encoder_strict_pairwise'],mats,[L,L,E]):
  for i,category in enumerate(categories):
   for j,model in enumerate(models):
    thematic_rows.append(dict(category=category,n_markers=int(counts[category]),metric=metric,model=model,accuracy_pct=mat[i,j]))
   thematic_rows.append(dict(category=category,n_markers=int(counts[category]),metric=metric,model='mean',accuracy_pct=mat[i].mean()))
 pd.DataFrame(thematic_rows).to_csv(out/'thematic_model_results.csv',index=False)
 fig,axs=plt.subplots(1,3,figsize=(11.5,4.25),gridspec_kw={'width_ratios':[9,9,6],'wspace':.09});table=pd.DataFrame({'category':categories,'n_markers':counts.values});cmap=plt.get_cmap('viridis')
 for j,(ax,mat,labels,title) in enumerate(zip(axs,mats,[SHORT[:8],SHORT[:8],SHORT[8:]],['LLM mean single-order','LLM order-robust','Encoder strict pairwise'])):
  means=mat.mean(axis=1);table[['llm_single_mean','llm_robust_mean','encoder_pairwise_mean'][j]]=means;mat=np.column_stack([mat,means]);im=ax.imshow(mat,vmin=0,vmax=100,cmap=cmap,aspect='auto');ax.set_xticks(range(len(labels)+1),labels+['Mean'],rotation=65,ha='right',rotation_mode='anchor',fontsize=9);ax.set_yticks(range(12),[f'{c.replace(" and " , " & ")} ({counts[c]})' for c in categories] if j==0 else []);ax.tick_params(length=0);ax.set_title(title,pad=12,weight='bold',fontsize=11);ax.axvline(len(labels)-.5,color='white',lw=2)
  for i in range(12):
   for k in range(mat.shape[1]):
    v=mat[i,k];ax.text(k,i,f'{v:.0f}' if k<len(labels) else f'{v:.1f}',ha='center',va='center',fontsize=9 if k<len(labels) else 8.5,weight='bold' if k==len(labels) else 'normal',color='white' if v<48 else '#151c28')
  for s in ax.spines.values():s.set_visible(False)
 fig.colorbar(im,ax=axs.tolist(),fraction=.018,pad=.02,label='Accuracy (%)');save(fig,'figure5_thematic_recognition');table.to_csv(out/'thematic_means.csv',index=False)
 print('Saved five PDF/PNG figure pairs, classification_transfer.csv, thematic_means.csv and thematic_model_results.csv to',out)
if __name__=='__main__':main()
