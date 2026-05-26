"""
OncoMark - SHAP Feature Importance Analysis
=============================================
Camellia Mazumder | S.N. Bose National Centre for Basic Sciences, 2024-2025

SHAP (SHapley Additive exPlanations) analysis for interpreting which genes
drive hallmark activity predictions in OncoMark.

This module computes:
    - SHAP values per hallmark output head
    - Top-N most influential genes per hallmark
    - Summary plots and beeswarm visualizations
    - Cross-hallmark feature overlap analysis

Reference: Lundberg & Lee, NeurIPS 2017 (SHAP)
Paper DOI: 10.1038/s42003-025-08727-z
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

HALLMARKS = ["AIM", "DCE", "EGS", "GIM", "RCD", "SPS", "AID", "IA", "ERI", "TPI"]

HALLMARK_FULL_NAMES = {
    "AIM": "Activating Invasion & Metastasis",
    "DCE": "Deregulating Cellular Energetics",
    "EGS": "Evading Growth Suppressors",
    "GIM": "Genome Instability & Mutation",
    "RCD": "Resisting Cell Death",
    "SPS": "Sustaining Proliferative Signaling",
    "AID": "Avoiding Immune Destruction",
    "IA":  "Inducing Angiogenesis",
    "ERI": "Enabling Replicative Immortality",
    "TPI": "Tumor-Promoting Inflammation",
}


def compute_shap_for_sklearn_model(
    model,
    X: np.ndarray,
    feature_names: list,
    hallmark: str,
    background_samples: int = 100,
    explain_samples: int = 200,
    random_state: int = 42
) -> tuple:
    """
    Compute SHAP values for a sklearn classifier on one hallmark.

    Args:
        model: fitted sklearn classifier
        X: feature matrix (n_samples, n_features)
        feature_names: gene names corresponding to columns
        hallmark: hallmark abbreviation (for labeling)
        background_samples: number of background samples for SHAP explainer
        explain_samples: number of samples to explain
        random_state: for reproducibility

    Returns:
        (shap_values, shap_df) where shap_df has mean |SHAP| per gene
    """
    if not SHAP_AVAILABLE:
        raise ImportError("Install shap: pip install shap")

    np.random.seed(random_state)

    # Sample background and explanation sets
    n_bg = min(background_samples, len(X))
    bg_idx = np.random.choice(len(X), size=n_bg, replace=False)
    background = X[bg_idx]

    n_exp = min(explain_samples, len(X))
    exp_idx = np.random.choice(len(X), size=n_exp, replace=False)
    X_explain = X[exp_idx]

    # Use TreeExplainer for tree-based models, KernelExplainer otherwise
    model_type = type(model).__name__
    try:
        if model_type in ('RandomForestClassifier', 'XGBClassifier',
                           'DecisionTreeClassifier', 'GradientBoostingClassifier'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_explain)
            # For binary classification, take class 1
            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba, background, silent=True
            )
            shap_values_raw = explainer.shap_values(X_explain, silent=True)
            sv = shap_values_raw[1] if isinstance(shap_values_raw, list) else shap_values_raw
    except Exception as e:
        print(f"SHAP computation failed for {model_type}: {e}")
        return None, None

    # Summarize: mean |SHAP| per gene
    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_df = pd.DataFrame({
        'gene': feature_names,
        'mean_abs_shap': mean_abs_shap,
        'hallmark': hallmark,
        'hallmark_full': HALLMARK_FULL_NAMES.get(hallmark, hallmark)
    }).sort_values('mean_abs_shap', ascending=False)

    return sv, shap_df


def plot_top_genes_per_hallmark(
    shap_results: dict,
    top_n: int = 20,
    output_path: str = 'shap_top_genes.png'
):
    """
    Bar plot of top-N most important genes for each hallmark.

    Args:
        shap_results: {hallmark: shap_df} from compute_shap_for_sklearn_model
        top_n: number of top genes to show
        output_path: save path
    """
    hallmarks = [h for h in shap_results if shap_results[h] is not None]
    n_hallmarks = len(hallmarks)

    if n_hallmarks == 0:
        print("No SHAP results to plot.")
        return

    cols = min(3, n_hallmarks)
    rows = (n_hallmarks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = np.array(axes).flatten() if n_hallmarks > 1 else [axes]

    for i, hallmark in enumerate(hallmarks):
        df = shap_results[hallmark].head(top_n)
        ax = axes[i]
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(df)))
        bars = ax.barh(range(len(df)), df['mean_abs_shap'].values, color=colors)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df['gene'].values, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP value|', fontsize=8)
        ax.set_title(
            f"{hallmark}\n{HALLMARK_FULL_NAMES.get(hallmark, '')}",
            fontsize=9, fontweight='bold'
        )
        ax.grid(axis='x', alpha=0.3)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(
        'OncoMark: Top Predictive Genes per Cancer Hallmark (SHAP Analysis)',
        fontsize=12, fontweight='bold', y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"SHAP top genes plot saved: {output_path}")
    return fig


def compute_hallmark_gene_overlap(shap_results: dict, top_n: int = 50) -> pd.DataFrame:
    """
    Compute overlap of top predictive genes between hallmarks.

    Shared gene signatures across hallmarks reflect biological
    interdependencies — a key finding in cancer hallmark research.

    Args:
        shap_results: {hallmark: shap_df}
        top_n: genes to consider per hallmark

    Returns:
        Overlap count matrix (n_hallmarks × n_hallmarks)
    """
    top_gene_sets = {}
    for hallmark, df in shap_results.items():
        if df is not None:
            top_gene_sets[hallmark] = set(df.head(top_n)['gene'].values)

    hallmarks = list(top_gene_sets.keys())
    overlap_matrix = np.zeros((len(hallmarks), len(hallmarks)), dtype=int)

    for i, h1 in enumerate(hallmarks):
        for j, h2 in enumerate(hallmarks):
            overlap_matrix[i, j] = len(top_gene_sets[h1] & top_gene_sets[h2])

    return pd.DataFrame(overlap_matrix, index=hallmarks, columns=hallmarks)


def plot_gene_overlap_heatmap(
    overlap_df: pd.DataFrame,
    output_path: str = 'gene_overlap_heatmap.png'
):
    """Plot hallmark-gene overlap as heatmap."""
    try:
        import seaborn as sns
    except ImportError:
        print("seaborn not available for heatmap. Skipping.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.zeros_like(overlap_df.values, dtype=bool)
    np.fill_diagonal(mask, True)  # hide diagonal (self-overlap)

    sns.heatmap(
        overlap_df,
        annot=True,
        fmt='d',
        cmap='Blues',
        mask=mask,
        ax=ax,
        linewidths=0.5,
        cbar_kws={'label': f'Shared top genes'}
    )
    ax.set_title(
        'Shared Predictive Genes Between Cancer Hallmarks\n'
        '(reflects biological pathway interdependencies)',
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Gene overlap heatmap saved: {output_path}")
    return fig


if __name__ == "__main__":
    print("OncoMark SHAP Analysis Demo")
    print("=" * 40)

    if not SHAP_AVAILABLE:
        print("Install shap to run this module: pip install shap")
        print("Demo will use mock SHAP values instead.\n")

    np.random.seed(42)
    n_samples, n_features = 200, 50
    gene_names = [f"GENE_{i:04d}" for i in range(n_features)]
    # Sprinkle in some real-ish gene names for demo
    known_genes = ["TP53", "KRAS", "EGFR", "MYC", "BRCA1", "PTEN",
                   "VHL", "APC", "RB1", "CDKN2A"]
    gene_names[:len(known_genes)] = known_genes
    X = np.random.randn(n_samples, n_features)

    # Mock SHAP results (since full OncoMark model requires the trained weights)
    print("Generating mock SHAP summary (install full OncoMark for real values)...")
    shap_results = {}
    for hallmark in HALLMARKS[:4]:  # demo with 4 hallmarks
        mean_abs = np.abs(np.random.randn(n_features))
        shap_df = pd.DataFrame({
            'gene': gene_names,
            'mean_abs_shap': mean_abs,
            'hallmark': hallmark,
            'hallmark_full': HALLMARK_FULL_NAMES.get(hallmark, hallmark)
        }).sort_values('mean_abs_shap', ascending=False)
        shap_results[hallmark] = shap_df

    # Plot
    plot_top_genes_per_hallmark(shap_results, top_n=10,
                                 output_path='shap_top_genes_demo.png')

    overlap = compute_hallmark_gene_overlap(shap_results, top_n=20)
    print("\nGene overlap matrix (top 20 genes):")
    print(overlap)
    plot_gene_overlap_heatmap(overlap, output_path='gene_overlap_demo.png')

    print("\nFor real SHAP analysis with the trained OncoMark model:")
    print("  pip install OncoMark shap")
    print("  from OncoMark import predict_hallmark_scores")
    print("  See: https://oncomark.readthedocs.io/en/latest/")
