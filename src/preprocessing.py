"""
OncoMark - Data Preprocessing Pipeline
=======================================
Camellia Mazumder | S.N. Bose National Centre for Basic Sciences, 2024-2025

Implements the data preprocessing workflow described in:
Priyadarshi et al., Communications Biology (Nature Portfolio), 2025
DOI: 10.1038/s42003-025-08727-z

Pipeline stages:
    1. Quality control filtering (mitochondrial content, gene count thresholds)
    2. Variance-based gene feature selection (top 9,326 genes)
    3. Rank-space transformation (batch effect mitigation)
    4. Log2 transformation + Z-score normalization
    5. Missing value imputation

Usage:
    from preprocessing import OncoMarkPreprocessor
    preprocessor = OncoMarkPreprocessor()
    X_processed = preprocessor.fit_transform(expression_df)
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class QualityControl:
    """
    Single-cell RNA-seq quality control filters.

    Thresholds follow the paper's methodology:
    - Mitochondrial content: exclude cells > 15%
    - Gene count: exclude cells with < 200 or > 6000 genes expressed
    """

    def __init__(
        self,
        min_genes: int = 200,
        max_genes: int = 6000,
        max_mito_pct: float = 0.15,
        mito_prefix: str = "MT-"
    ):
        self.min_genes = min_genes
        self.max_genes = max_genes
        self.max_mito_pct = max_mito_pct
        self.mito_prefix = mito_prefix

    def filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply QC filters to a gene expression DataFrame.

        Args:
            df: DataFrame of shape (n_cells, n_genes), columns = gene names

        Returns:
            Filtered DataFrame
        """
        original_count = len(df)

        # Gene count filter
        gene_counts = (df > 0).sum(axis=1)
        mask_genes = (gene_counts >= self.min_genes) & (gene_counts <= self.max_genes)

        # Mitochondrial content filter
        mito_genes = [g for g in df.columns if g.startswith(self.mito_prefix)]
        if mito_genes:
            mito_counts = df[mito_genes].sum(axis=1)
            total_counts = df.sum(axis=1)
            mito_pct = mito_counts / total_counts.replace(0, np.nan)
            mask_mito = mito_pct <= self.max_mito_pct
            combined_mask = mask_genes & mask_mito
        else:
            combined_mask = mask_genes

        filtered_df = df[combined_mask]
        removed = original_count - len(filtered_df)
        print(f"QC filtering: {original_count} → {len(filtered_df)} cells "
              f"({removed} removed, {removed/original_count*100:.1f}%)")

        return filtered_df


class FeatureSelector:
    """
    Variance-based gene feature selection.

    Selects the top-k highest-variance genes, intersecting across
    hallmark-positive and hallmark-negative sample sets (as per paper).
    Paper uses top 9,326 genes after intersection.
    """

    def __init__(self, n_top_genes: int = 9326):
        self.n_top_genes = n_top_genes
        self.selected_genes_ = None

    def fit(self, df_positive: pd.DataFrame, df_negative: pd.DataFrame) -> "FeatureSelector":
        """
        Identify top variable genes, intersected across both classes.

        Args:
            df_positive: hallmark-positive pseudo-bulk samples
            df_negative: hallmark-negative pseudo-bulk samples
        """
        # Compute per-gene variance, exclude zero/undefined variance
        var_pos = df_positive.var(axis=0).replace(0, np.nan).dropna()
        var_neg = df_negative.var(axis=0).replace(0, np.nan).dropna()

        # Top-k genes from each class
        top_pos = set(var_pos.nlargest(self.n_top_genes).index)
        top_neg = set(var_neg.nlargest(self.n_top_genes).index)

        # Intersection
        self.selected_genes_ = sorted(top_pos & top_neg)
        print(f"Feature selection: {len(self.selected_genes_)} genes selected "
              f"(intersection of top {self.n_top_genes} from each class)")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.selected_genes_ is None:
            raise ValueError("Call fit() before transform()")
        available = [g for g in self.selected_genes_ if g in df.columns]
        missing = len(self.selected_genes_) - len(available)
        if missing > 0:
            print(f"  Note: {missing} selected genes not found in input — will be imputed as 0")
        result = df.reindex(columns=self.selected_genes_, fill_value=0)
        return result

    def fit_transform(self, df_positive: pd.DataFrame, df_negative: pd.DataFrame) -> tuple:
        self.fit(df_positive, df_negative)
        return self.transform(df_positive), self.transform(df_negative)


class RankNormalizer:
    """
    Rank-space transformation for batch effect mitigation.

    Converts expression values within each sample to fractional ranks,
    minimizing technical variation from different sequencing platforms
    and library preparation protocols. Applied prior to log2 + z-score.
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply rank normalization sample-wise (row-wise).

        Args:
            df: DataFrame of shape (n_samples, n_genes)

        Returns:
            Rank-normalized DataFrame (values in [0, 1])
        """
        ranked = df.rank(axis=1, method='average', na_option='keep')
        # Normalize to [0, 1]
        n_genes = df.shape[1]
        ranked_normalized = ranked / n_genes
        return ranked_normalized


class OncoMarkPreprocessor:
    """
    Full preprocessing pipeline for OncoMark inference.

    Stages (in order):
    1. Missing value imputation (fill with 0)
    2. Rank-space transformation
    3. Log2 transformation: log2(x + 1)
    4. Z-score normalization (per-gene, mean=0, std=1)

    Note: For training from scratch, use QualityControl and FeatureSelector
    first, then pass the result to this pipeline.
    """

    def __init__(self):
        self.rank_normalizer = RankNormalizer()
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "OncoMarkPreprocessor":
        """Fit scaler on training data."""
        X = self._rank_log_transform(df)
        self.scaler.fit(X)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler."""
        if not self._fitted:
            raise ValueError("Call fit() before transform()")
        X = self._rank_log_transform(df)
        return self.scaler.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        X = self._rank_log_transform(df)
        self._fitted = True
        return self.scaler.fit_transform(X)

    def _rank_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Internal: impute → rank → log2."""
        # Step 1: impute missing as 0
        X = df.fillna(0)
        # Step 2: rank normalization
        X = self.rank_normalizer.transform(X)
        # Step 3: log2 transform
        X = np.log2(X + 1)
        return X


def load_and_validate_input(filepath: str, sep: str = ",") -> pd.DataFrame:
    """
    Load a gene expression CSV and run basic validation.

    Expected format:
        - Rows = samples
        - Columns = gene names (HGNC symbols preferred)
        - Values = raw or normalized read counts

    Args:
        filepath: path to CSV file
        sep: delimiter (default: comma)

    Returns:
        Validated DataFrame
    """
    df = pd.read_csv(filepath, index_col=0, sep=sep)
    print(f"Loaded: {df.shape[0]} samples × {df.shape[1]} genes")

    # Check for negative values
    if (df < 0).any().any():
        raise ValueError("Expression matrix contains negative values. "
                         "Expected raw counts or non-negative normalized values.")

    # Warn if very few genes
    if df.shape[1] < 1000:
        warnings.warn(f"Only {df.shape[1]} genes detected. "
                      "OncoMark expects at least ~9,326 genes for full accuracy.")

    # Basic stats
    print(f"  Value range: [{df.min().min():.2f}, {df.max().max():.2f}]")
    print(f"  Sparsity: {(df == 0).sum().sum() / df.size * 100:.1f}% zeros")

    return df


if __name__ == "__main__":
    # Example usage with synthetic data
    print("OncoMark Preprocessing Pipeline Demo")
    print("=" * 45)

    # Generate synthetic expression data
    np.random.seed(42)
    n_samples, n_genes = 50, 500
    gene_names = [f"GENE_{i}" for i in range(n_genes)]
    # Add some fake MT- genes
    gene_names[:5] = [f"MT-{i}" for i in range(5)]

    raw_data = pd.DataFrame(
        np.random.negative_binomial(10, 0.3, size=(n_samples, n_genes)),
        columns=gene_names
    )
    print(f"\nInput: {raw_data.shape[0]} samples × {raw_data.shape[1]} genes")

    # Run QC
    qc = QualityControl()
    filtered = qc.filter(raw_data)

    # Run full preprocessing
    preprocessor = OncoMarkPreprocessor()
    X_processed = preprocessor.fit_transform(filtered)

    print(f"\nOutput shape: {X_processed.shape}")
    print(f"Mean (should be ~0): {X_processed.mean():.4f}")
    print(f"Std  (should be ~1): {X_processed.std():.4f}")
    print("\nPreprocessing complete. Ready for OncoMark inference.")
