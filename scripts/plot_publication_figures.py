"""Rebuild publication figures from validated analysis CSVs (no inference).

Run after the existing full-inversion analysis script. All outputs go to
results/analysis/ and use manuscript-compatible filenames.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODELS = ["gemini", "claude", "gpt54", "gemma4_12b_it", "mistral_nemo_12b", "qwen3_8b", "aya_expanse_8b", "llama31_8b"]
LABELS = {
    'gemini': 'Gemini 3.8 Flash',
    'claude': 'Claude Opus 4.8',
    'gpt54': 'GPT-5.4',
    'gemma4_12b_it': 'Gemma 4 12B IT',
    'mistral_nemo_12b': 'Mistral NeMo 12B Instruct',
    'qwen3_8b': 'Qwen3 8B',
    'aya_expanse_8b': 'Aya Expanse 8B',
    'llama31_8b': 'Llama 3.1 8B Instruct',
}
CATEGORIES = [
    "Sports and Recreation", "Food and Hospitality", "Health and Medicine",
    "Science and Nature", "Technology and Communications", "Construction and Tools",
    "Public Administration and Law", "Business, Commerce and Finance",
    "Education and Social Services", "Transportation and Vehicles",
    "Road Use and Safety", "Everyday Life and Consumer Goods",
]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42, "savefig.dpi": 240})


def _load(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run the full analysis first")
    df = pd.read_csv(path)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} lacks columns: {sorted(missing)}")
    return df


def _check_models(df: pd.DataFrame, name: str) -> pd.DataFrame:
    present = set(df.model)
    if present != set(MODELS):
        raise ValueError(f"{name}: expected 8 models, found missing={set(MODELS)-present}, extra={present-set(MODELS)}")
    return df


def _save(fig, directory: Path, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.12, dpi=240)
    plt.close(fig)


def plot_accuracy_and_reversal(overall: pd.DataFrame, out: Path) -> None:
    arr = overall.set_index("model").loc[MODELS]
    original = arr.original.to_numpy(float)*100
    both = arr.correct_both.to_numpy(float)*100
    one = arr.correct_one.to_numpy(float)*100
    neither = arr.incorrect_both.to_numpy(float)*100
    if not np.allclose(both+one+neither,100,atol=.01):
        raise ValueError("Two-order results do not sum to 100%")
    fig, a = plt.subplots(figsize=(8.5, 5.6))
    a.plot([0,100],[0,100],"--",lw=1.2,color="#999999")
    for i in range(8):
        color="#31688e" if i<3 else "#8051a1"
        a.scatter(original[i],both[i],s=115,c=color,edgecolors="white",linewidths=1,zorder=3)
        a.text(original[i],both[i],str(i+1),ha="center",va="center",color="white",fontsize=7.5,weight="bold",zorder=4)
    a.set(
        xlim=(0, 102),
        ylim=(0, 102),
        xlabel="Original-order accuracy (%)",
        ylabel="Order-robust accuracy (%)"
    )
    a.set_xticks(np.arange(0, 101, 20))
    a.set_yticks(np.arange(0, 101, 20))
    a.set_aspect("equal", adjustable="box")
    a.set_title("Original vs. order-robust accuracy",weight="bold")
    a.grid(alpha=.17)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="none", color="#31688e" if i<3 else "#8051a1", label=f"{i+1}. {LABELS[m]}") for i,m in enumerate(MODELS)]
    a.legend(handles=handles, loc="center left", bbox_to_anchor=(1.03,.5), frameon=False, fontsize=8)
    _save(fig,out,"original_vs_robust_accuracy")
    fig, b = plt.subplots(figsize=(10.5, 5.2))
    y=np.arange(8)
    b.barh(y,both,color="#287d80",label="Correct in both orders")
    b.barh(y,one,left=both,color="#d5a04b",label="Correct in one order")
    b.barh(y,neither,left=both+one,color="#bf696a",label="Incorrect in both orders")
    b.set(yticks=y,yticklabels=[f"{i+1}. {LABELS[m]}" for i,m in enumerate(MODELS)],xlim=(0,100),xlabel="Share of sentence pairs (%)")
    b.invert_yaxis();b.grid(axis="x",alpha=.17);b.set_axisbelow(True)
    for i in y:
        for value,left in ((both[i],0),(one[i],both[i]),(neither[i],both[i]+one[i])):
            if value>=10: b.text(left+value/2,i,f"{value:.1f}",ha="center",va="center",fontsize=7.7)
    b.set_title("Outcomes under candidate reversal",weight="bold")
    b.legend(loc="upper center",bbox_to_anchor=(.5,-.14),ncol=3,frameon=False,fontsize=8.5)
    _save(fig,out,"order_robustness")


def plot_ci(ci: pd.DataFrame, out: Path) -> None:
    df=ci.loc[ci.metric.eq("correct_both")].set_index("model").loc[MODELS]
    if len(df)!=8: raise ValueError("Missing robust CIs")
    means=df.estimate.to_numpy(float)*100
    low=df.low95.to_numpy(float)*100
    high=df.high95.to_numpy(float)*100
    if np.any(low>means) or np.any(means>high):raise ValueError("Invalid confidence intervals")
    fig,ax=plt.subplots(figsize=(9.5,4.1))
    y=np.arange(8)
    for i in y:
        ax.errorbar(means[i],i,xerr=np.array([[means[i]-low[i]],[high[i]-means[i]]]),fmt="o",capsize=3,ms=6,elinewidth=2,color="#31688e" if i<3 else "#8051a1")
        ax.text(104,i,f"{means[i]:.2f}% [{low[i]:.2f}, {high[i]:.2f}]",va="center",fontsize=8.5)
    ax.set(yticks=y,yticklabels=[LABELS[m] for m in MODELS],xlim=(0,150),xlabel="Order-robust accuracy (%)")
    ax.invert_yaxis();ax.set_xticks([0,20,40,60,80,100]);ax.grid(axis="x",alpha=.16)
    ax.set_title("Order-robust accuracy and 95% bootstrap CIs",weight="bold")
    _save(fig,out,"robust_accuracy_ci")


def aggregate_categories(pairs: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Map each source first/top-level domain to one broad category; pool pair outcomes."""
    if mapping.primary_domain.duplicated().any():raise ValueError("Duplicate primary domains in mapping")
    domain=pairs.domaine.fillna("").astype(str).str.split(";",regex=False).str[0].str.split(">",regex=False).str[0].str.strip()
    lookup=dict(zip(mapping.primary_domain,mapping.broad_category))
    missing=sorted(set(domain)-set(lookup))
    if missing: raise ValueError(f"Unmapped primary domains: {missing}. Update data/domain_to_broad_category.csv")
    df=pairs.assign(primary_domain=domain,broad_category=domain.map(lookup))
    if len(df)!=1205*8 or df.feature_id.nunique()!=241:raise ValueError("Expected 1205 pairs x 8 models and 241 markers")
    summary=(df.groupby(["model","broad_category"],as_index=False)
        .agg(n_pairs=("pair_id","size"),n_markers=("feature_id","nunique"),robust_successes=("correct_both","sum")))
    summary["robust"]=summary.robust_successes/summary.n_pairs
    for model, part in summary.groupby("model"):
        if part.n_pairs.sum()!=1205 or part.n_markers.sum()!=241 or set(part.broad_category)!=set(CATEGORIES):
            raise ValueError(f"Incomplete category coverage for {model}")
    return summary


def plot_heatmap(data: pd.DataFrame,out:Path) -> None:
    d=data.set_index(["model","broad_category"])
    matrix=np.array([[100*float(d.loc[(model,cat),"robust"]) for model in MODELS] for cat in CATEGORIES])
    counts=[int(d.loc[(MODELS[0],cat),"n_markers"]) for cat in CATEGORIES]
    fig,ax=plt.subplots(figsize=(13.6,7.3))
    im=ax.imshow(matrix,cmap="viridis",vmin=0,vmax=100,aspect="auto")
    ax.set_xticks(np.arange(8),[LABELS[m] for m in MODELS],rotation=30,ha="right",rotation_mode="anchor",fontsize=9)
    ax.set_yticks(np.arange(12),[f"{cat} (n={counts[i]})" for i,cat in enumerate(CATEGORIES)],fontsize=9)
    for j in range(12):
        for i in range(8):
            val=matrix[j,i]
            ax.text(i,j,f"{val:.0f}",ha="center",va="center",fontsize=8.1,color="white" if val<48 else "#151c28")
    ax.tick_params(length=0)
    ax.set_title("Order-robust accuracy by terminological category",weight="bold",pad=14)
    cb=fig.colorbar(im,ax=ax,fraction=.025,pad=.02);cb.set_label("Order-robust accuracy (%)")
    fig.tight_layout()
    _save(fig,out,"domain_heatmap")


def generate_figures(out: Path, mapping_path: Path) -> None:
    """Call this from the bottom of the full-inversion analyzer after CSV export."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    overall=_check_models(_load(out/"overall.csv",{"model","original","correct_both","correct_one","incorrect_both"}),"overall")
    ci=_check_models(_load(out/"cluster_bootstrap_ci.csv",{"model","metric","estimate","low95","high95"}),"CI")
    pairs=_check_models(_load(out/"paired_observations.csv",{"model","pair_id","feature_id","domaine","correct_both"}),"paired observations")
    mapping=_load(Path(mapping_path),{"primary_domain","broad_category"})
    plot_accuracy_and_reversal(overall,out)
    plot_ci(ci,out)
    categories=aggregate_categories(pairs,mapping)
    categories.to_csv(out/"broad_category_model_results.csv",index=False)
    plot_heatmap(categories,out)
    print("Saved:",*[str(out/f"{name}.{ext}") for name in ("original_vs_robust_accuracy","order_robustness","robust_accuracy_ci","domain_heatmap") for ext in ("png","pdf")],sep="\n  ")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=Path("results/analysis"))
    parser.add_argument("--mapping",type=Path,default=Path("data/domain_to_broad_category.csv"))
    args=parser.parse_args()
    generate_figures(args.out,args.mapping)

if __name__=="__main__":main()
