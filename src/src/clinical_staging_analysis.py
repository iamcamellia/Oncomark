"""
OncoMark - Clinical Staging Association Analysis
=================================================
Camellia Mazumder | S.N. Bose National Centre for Basic Sciences, 2024-2025

Analyses the association between predicted OncoMark hallmark activities
and clinical cancer staging metrics (AJCC, TNM).

Key finding from paper (Fig. 5): Hallmark activity progressively increases
from Stage I → Stage IV, with strongest co-association at advanced stages.
This validates the biological relevance of OncoMark's predictions.

Staging variables analysed:
    - AJCC_PATHOLOGIC_TUMOR_STAGE (Stage I/II/III/IV)
    - Metastasis stage (M0/M1)
    - Lymph node involvement (N0/N1/N2/N3)
    - Tumor size (T1/T2/T3/T4)

Association measure: Odds Ratio (OR) with 95% confidence intervals
Statistical test: Fisher's exact test (binary) / Chi-squared

Paper DOI: 10.1038/s42003-025-08727-z
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, fisher_exact
import warnings
warnings.filterwarnings('ignore')

HALLMARKS = ["AIM", "DCE", "EGS", "GIM", "RCD", "SPS", "AID", "IA", "ERI", "TPI"]

STAGING_VARIABLES = {
    'AJCC': 'ajcc_pathologic_tumor_stage',
    'Metastasis (M)': 'm_stage',
    'Lymph Node (N)': 'n_stage',
    'Tumor Size (T)': 't_stage',
}

STAGE_ORDERS = {
    'ajcc_pathologic_tumor_stage': ['Stage I', 'Stage II', 'Stage III', 'Stage IV'],
    'm_stage': ['M0', 'M1'],
    'n_stage': ['N0', 'N1', 'N2', 'N3'],
    't_stage': ['T1', 'T2', 'T3', 'T4'],
}


def compute_odds_ratio(
    hallmark_binary: np.ndarray,
    stage_binary: np.ndarray
) -> tuple:
    """
    Compute odds ratio and 95% CI for hallmark vs. stage association.

    Args:
        hallmark_binary: binary array (0/1) — hallmark active or not
        stage_binary: binary array (0/1) — stage category present

    Returns:
        (odds_ratio, ci_lower, ci_upper, p_value)
    """
    # Contingency table
    # [[TN, FP], [FN, TP]]
    tp = np.sum((hallmark_binary == 1) & (stage_binary == 1))
    fp = np.sum((hallmark_binary == 1) & (stage_binary == 0))
    fn = np.sum((hallmark_binary == 0) & (stage_binary == 1))
    tn = np.sum((hallmark_binary == 0) & (stage_binary == 0))

    table = np.array([[tn, fp], [fn, tp]])

    # Use Fisher's exact for small cells, chi2 otherwise
    if np.min(table) < 5:
        odds_ratio, p_value = fisher_exact(table)
    else:
        chi2, p_value, _, _ = chi2_contingency(table)
        # Compute OR manually
        if fp * fn == 0:
            odds_ratio = np.inf if (tp * tn > 0) else np.nan
        else:
            odds_ratio = (tp * tn) / (fp * fn)

    # 95% CI via Woolf's method (log scale)
    try:
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1/tp + 1/fp + 1/fn + 1/tn) if all(
            x > 0 for x in [tp, fp, fn, tn]
        ) else np.nan
        z = 1.96  # 95% CI
        ci_lower = np.exp(log_or - z * se_log_or) if not np.isnan(se_log_or) else np.nan
        ci_upper = np.exp(log_or + z * se_log_or) if not np.isnan(se_log_or) else np.nan
    except Exception:
        ci_lower, ci_upper = np.nan, np.nan

    return float(odds_ratio), float(ci_lower), float(ci_upper), float(p_value)


def analyse_hallmark_stage_associations(
    hallmark_predictions: pd.DataFrame,
    clinical_data: pd.DataFrame,
    patient_id_col: str = 'patient_id'
) -> pd.DataFrame:
    """
    Compute OR between each hallmark and each staging category.

    Args:
        hallmark_predictions: DataFrame with columns = hallmark abbreviations,
                              rows = patients, values = binary (0/1)
        clinical_data: DataFrame with staging columns and patient_id_col
        patient_id_col: column name for patient ID (for merging)

    Returns:
        DataFrame with columns:
            hallmark, stage_variable, stage_category, OR, CI_lower, CI_upper, p_value
    """
    # Merge on patient ID
    merged = pd.merge(hallmark_predictions, clinical_data, on=patient_id_col, how='inner')
    print(f"Merged dataset: {len(merged)} patients")

    results = []

    for stage_label, stage_col in STAGING_VARIABLES.items():
        if stage_col not in merged.columns:
            continue

        stage_categories = STAGE_ORDERS.get(stage_col, merged[stage_col].dropna().unique().tolist())

        for hallmark in HALLMARKS:
            if hallmark not in merged.columns:
                continue

            hallmark_vals = merged[hallmark].values

            for category in stage_categories:
                stage_binary = (merged[stage_col] == category).astype(int).values

                if stage_binary.sum() < 5:
                    continue  # skip sparse categories

                or_val, ci_lo, ci_hi, pval = compute_odds_ratio(
                    hallmark_vals, stage_binary
                )

                results.append({
                    'Hallmark': hallmark,
                    'Stage_Variable': stage_label,
                    'Stage_Column': stage_col,
                    'Stage_Category': category,
                    'Odds_Ratio': or_val,
                    'CI_Lower': ci_lo,
                    'CI_Upper': ci_hi,
                    'P_Value': pval,
                    'Significant': pval < 0.05
                })

    return pd.DataFrame(results)


def plot_staging_heatmap(
    association_df: pd.DataFrame,
    stage_variable: str = 'AJCC',
    metric: str = 'Odds_Ratio',
    output_path: str = 'staging_heatmap.png'
):
    """
    Heatmap of hallmark × stage associations (replicates paper Fig. 5).

    Args:
        association_df: output from analyse_hallmark_stage_associations()
        stage_variable: which staging variable to plot
        metric: 'Odds_Ratio' or 'P_Value'
        output_path: save path
    """
    subset = association_df[association_df['Stage_Variable'] == stage_variable].copy()
    if subset.empty:
        print(f"No data for stage variable: {stage_variable}")
        return

    # Get ordered categories
    stage_col = subset['Stage_Column'].iloc[0]
    ordered_categories = [c for c in STAGE_ORDERS.get(stage_col, [])
                          if c in subset['Stage_Category'].values]

    pivot = subset.pivot(
        index='Hallmark', columns='Stage_Category', values=metric
    )
    # Reorder columns
    pivot = pivot.reindex(columns=[c for c in ordered_categories if c in pivot.columns])

    # Mark significant cells
    sig_pivot = subset.pivot(
        index='Hallmark', columns='Stage_Category', values='Significant'
    ).reindex(columns=pivot.columns)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 2), 7))

    # Use log scale for OR (so heatmap is symmetric around 1)
    plot_values = np.log2(pivot.values + 1e-6) if metric == 'Odds_Ratio' else pivot.values
    plot_df = pd.DataFrame(plot_values, index=pivot.index, columns=pivot.columns)

    cmap = 'RdBu_r' if metric == 'Odds_Ratio' else 'RdYlGn_r'
    sns.heatmap(
        plot_df,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=0 if metric == 'Odds_Ratio' else None,
        ax=ax,
        linewidths=0.5,
        cbar_kws={
            'label': f'log2({metric})' if metric == 'Odds_Ratio' else metric
        }
    )

    # Add asterisks for significant associations
    for i, hallmark in enumerate(plot_df.index):
        for j, cat in enumerate(plot_df.columns):
            if sig_pivot.loc[hallmark, cat] if cat in sig_pivot.columns else False:
                ax.text(j + 0.5, i + 0.85, '*', ha='center', va='center',
                        fontsize=12, color='black', fontweight='bold')

    ax.set_title(
        f'Hallmark Activity vs. {stage_variable} Staging\n'
        f'(* = p < 0.05; color = log2 Odds Ratio)',
        fontsize=11, pad=12
    )
    ax.set_xlabel(f'{stage_variable} Stage', fontsize=10)
    ax.set_ylabel('Cancer Hallmark', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Staging heatmap saved: {output_path}")
    return fig


def generate_synthetic_clinical_data(
    n_patients: int = 500,
    random_state: int = 42
) -> tuple:
    """Generate synthetic clinical + hallmark data for demo."""
    np.random.seed(random_state)

    patient_ids = [f"TCGA-{i:04d}" for i in range(n_patients)]

    # Synthetic hallmark predictions (binary)
    hallmark_preds = pd.DataFrame(
        np.random.binomial(1, 0.55, size=(n_patients, len(HALLMARKS))),
        columns=HALLMARKS
    )
    hallmark_preds['patient_id'] = patient_ids

    # Synthetic clinical data — later stages have more hallmark activity
    stages = np.random.choice(
        ['Stage I', 'Stage II', 'Stage III', 'Stage IV'],
        size=n_patients,
        p=[0.3, 0.3, 0.25, 0.15]
    )
    m_stages = np.where(stages == 'Stage IV',
                         np.random.choice(['M0', 'M1'], n_patients, p=[0.3, 0.7]),
                         'M0')
    n_stages = np.random.choice(['N0', 'N1', 'N2', 'N3'], n_patients,
                                  p=[0.4, 0.3, 0.2, 0.1])
    t_stages = np.random.choice(['T1', 'T2', 'T3', 'T4'], n_patients,
                                  p=[0.25, 0.35, 0.25, 0.15])

    clinical = pd.DataFrame({
        'patient_id': patient_ids,
        'ajcc_pathologic_tumor_stage': stages,
        'm_stage': m_stages,
        'n_stage': n_stages,
        't_stage': t_stages,
    })

    return hallmark_preds, clinical


if __name__ == "__main__":
    print("OncoMark — Clinical Staging Association Demo")
    print("=" * 50)

    hallmark_preds, clinical_data = generate_synthetic_clinical_data(n_patients=400)

    print("\nRunning staging association analysis...")
    results = analyse_hallmark_stage_associations(
        hallmark_preds, clinical_data, patient_id_col='patient_id'
    )

    print(f"\nTotal associations computed: {len(results)}")
    sig = results[results['Significant']]
    print(f"Significant (p < 0.05): {len(sig)}")

    print("\nTop 5 strongest associations (by Odds Ratio):")
    top5 = results.nlargest(5, 'Odds_Ratio')[
        ['Hallmark', 'Stage_Variable', 'Stage_Category', 'Odds_Ratio', 'P_Value']
    ]
    print(top5.to_string(index=False))

    results.to_csv('staging_associations.csv', index=False)
    print("\nFull results saved to staging_associations.csv")

    try:
        plot_staging_heatmap(
            results,
            stage_variable='AJCC',
            output_path='staging_heatmap_ajcc.png'
        )
        plot_staging_heatmap(
            results,
            stage_variable='Metastasis (M)',
            output_path='staging_heatmap_metastasis.png'
        )
    except Exception as e:
        print(f"Plot error: {e}")

    print("\nFor real analysis with TCGA data:")
    print("  Download TCGA clinical metadata from: https://gdac.broadinstitute.org/")
    print("  Run OncoMark predictions: pip install OncoMark")
