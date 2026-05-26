"""
OncoMark - ML Baseline Benchmarking
=====================================
Camellia Mazumder | S.N. Bose National Centre for Basic Sciences, 2024-2025

Benchmarks classical supervised ML models against the OncoMark multi-task
neural network for cancer hallmark prediction.

Baselines evaluated (as described in paper Supplementary Fig. 6):
    - Logistic Regression (LR)
    - Support Vector Classifier (SVC)
    - Decision Tree (DT)
    - Random Forest (RF)
    - XGBoost
    - Multi-Layer Perceptron (MLP)

Key finding from paper: all baselines exhibited strong bias toward predicting
near-zero hallmark probabilities, failing to learn meaningful cancer patterns.
OncoMark's MTL architecture overcame this by learning shared representations
across all hallmarks simultaneously.

DOI: 10.1038/s42003-025-08727-z
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed. Skipping XGBoost baseline.")


# Cancer hallmarks (abbreviations from paper)
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


def get_baseline_models() -> dict:
    """Return dictionary of baseline classifiers."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        "SVC": SVC(
            probability=True, random_state=42, class_weight='balanced'
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=10, random_state=42, class_weight='balanced'
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42,
            class_weight='balanced', n_jobs=-1
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        ),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, max_depth=6,
            random_state=42, eval_metric='logloss',
            use_label_encoder=False, n_jobs=-1
        )
    return models


def evaluate_model_single_hallmark(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 2
) -> dict:
    """
    Evaluate a single model on one hallmark with stratified k-fold CV.

    Mirrors the paper's evaluation protocol: 5-fold CV repeated twice.

    Args:
        model: sklearn-compatible classifier
        X: feature matrix (n_samples, n_features)
        y: binary labels (0/1)
        n_splits: number of CV folds (default: 5, as in paper)
        n_repeats: number of repetitions (default: 2, as in paper)

    Returns:
        dict of metric arrays across folds
    """
    metrics = {
        'accuracy': [], 'f1': [], 'precision': [],
        'recall': [], 'roc_auc': [], 'balanced_accuracy': []
    }

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=42 + repeat)
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if hasattr(model, 'predict_proba'):
                y_prob = model.predict_proba(X_test)[:, 1]
                metrics['roc_auc'].append(roc_auc_score(y_test, y_prob))
            else:
                metrics['roc_auc'].append(np.nan)

            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics['balanced_accuracy'].append(balanced_accuracy_score(y_test, y_pred))

    return {k: np.array(v) for k, v in metrics.items()}


def run_full_benchmark(
    X: np.ndarray,
    y_dict: dict,
    hallmarks: list = None,
    n_splits: int = 5,
    n_repeats: int = 2
) -> pd.DataFrame:
    """
    Benchmark all baseline models across all hallmarks.

    Args:
        X: feature matrix (n_samples, n_features)
        y_dict: {hallmark_name: binary_label_array}
        hallmarks: subset of hallmarks to evaluate (default: all)
        n_splits, n_repeats: CV parameters

    Returns:
        DataFrame with mean ± std metrics per model × hallmark
    """
    if hallmarks is None:
        hallmarks = list(y_dict.keys())

    models = get_baseline_models()
    results = []

    total = len(models) * len(hallmarks)
    done = 0

    for model_name, model in models.items():
        for hallmark in hallmarks:
            if hallmark not in y_dict:
                continue

            y = y_dict[hallmark]

            # Skip if only one class
            if len(np.unique(y)) < 2:
                print(f"  Skipping {model_name} × {hallmark}: only one class in labels")
                continue

            fold_metrics = evaluate_model_single_hallmark(
                model, X, y, n_splits=n_splits, n_repeats=n_repeats
            )

            row = {
                'Model': model_name,
                'Hallmark': hallmark,
                'Hallmark_Full': HALLMARK_FULL_NAMES.get(hallmark, hallmark),
            }
            for metric, values in fold_metrics.items():
                row[f'{metric}_mean'] = np.nanmean(values)
                row[f'{metric}_std'] = np.nanstd(values)

            results.append(row)
            done += 1
            print(f"  [{done}/{total}] {model_name} × {hallmark}: "
                  f"F1={row['f1_mean']:.4f} ± {row['f1_std']:.4f}, "
                  f"AUC={row['roc_auc_mean']:.4f}")

    return pd.DataFrame(results)


def plot_benchmark_heatmap(
    results_df: pd.DataFrame,
    metric: str = 'f1_mean',
    output_path: str = 'benchmark_heatmap.png'
):
    """
    Plot heatmap of model performance across hallmarks.

    Args:
        results_df: output from run_full_benchmark()
        metric: column to plot (default: f1_mean)
        output_path: where to save the figure
    """
    pivot = results_df.pivot(index='Model', columns='Hallmark', values=metric)

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='YlOrRd',
        vmin=0.5,
        vmax=1.0,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': metric.replace('_', ' ').title()}
    )
    ax.set_title(
        f'Baseline Model Benchmarking — {metric.replace("_", " ").title()}\n'
        f'(OncoMark MTL framework outperforms all baselines — '
        f'see paper Fig. S6)',
        fontsize=11, pad=15
    )
    ax.set_xlabel('Cancer Hallmark')
    ax.set_ylabel('Model')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved: {output_path}")
    return fig


def generate_synthetic_benchmark_data(
    n_samples: int = 500,
    n_features: int = 200,
    n_hallmarks: int = 10,
    random_state: int = 42
) -> tuple:
    """
    Generate synthetic data for benchmarking demonstration.

    Returns:
        X: feature matrix
        y_dict: {hallmark: labels}
    """
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)

    y_dict = {}
    for i, hallmark in enumerate(HALLMARKS[:n_hallmarks]):
        # Simulate hallmark labels with some signal from features
        weights = np.random.randn(n_features) * 0.1
        logits = X @ weights + np.random.randn(n_samples) * 0.5
        probs = 1 / (1 + np.exp(-logits))
        y_dict[hallmark] = (probs > 0.5).astype(int)

    return X, y_dict


if __name__ == "__main__":
    print("OncoMark — Baseline ML Benchmarking Demo")
    print("=" * 50)
    print("Generating synthetic data for demonstration...")

    X, y_dict = generate_synthetic_benchmark_data(
        n_samples=300, n_features=100, n_hallmarks=3
    )
    hallmarks_to_test = list(y_dict.keys())

    print(f"\nData: {X.shape[0]} samples × {X.shape[1]} features")
    for h in hallmarks_to_test:
        pos = y_dict[h].sum()
        print(f"  {h}: {pos} positive / {len(y_dict[h]) - pos} negative")

    print("\nRunning 5-fold CV × 2 repeats for each model × hallmark...")
    results = run_full_benchmark(X, y_dict, hallmarks=hallmarks_to_test,
                                  n_splits=5, n_repeats=2)

    print("\n--- Summary ---")
    summary = results.groupby('Model')['f1_mean'].agg(['mean', 'std'])
    print(summary.sort_values('mean', ascending=False).round(4))

    # Save results
    results.to_csv('benchmark_results.csv', index=False)
    print("\nResults saved to benchmark_results.csv")

    # Plot
    try:
        plot_benchmark_heatmap(results, output_path='benchmark_heatmap.png')
    except Exception as e:
        print(f"Plot skipped: {e}")
