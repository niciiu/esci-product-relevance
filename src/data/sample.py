"""
Stratified sampling utilities for the ESCI dataset.

Sampling here is always applied to ONE side of the official split at a
time (official TRAIN or official TEST) so the two populations never mix.
The caller is responsible for passing in already-split data (see
src/data/split.py).
"""

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

DEFAULT_LABEL_COLUMN = "esci_label"
DEFAULT_SEED = 42
ESCI_LABELS = ("E", "S", "C", "I")


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_sampling_input(df: pd.DataFrame, n: int, label_col: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
    if len(df) == 0:
        raise ValueError("Cannot sample from an empty DataFrame.")
    if n <= 0:
        raise ValueError("Sample size n must be greater than 0.")
    if n > len(df):
        raise ValueError(f"Sample size n={n:,} cannot exceed population size={len(df):,}.")
    if df[label_col].isna().any():
        raise ValueError(f"Label column '{label_col}' contains missing values.")

    unknown_labels = set(df[label_col].unique()) - set(ESCI_LABELS)
    if unknown_labels:
        raise ValueError(
            f"Label column contains unsupported ESCI labels: {sorted(unknown_labels)}. "
            f"Expected only {list(ESCI_LABELS)}."
        )


def _allocate_sample_counts(df: pd.DataFrame, n: int, label_col: str) -> dict:
    """Allocate exactly n samples across classes proportionally (largest remainder method)."""
    class_counts = df[label_col].value_counts().sort_index()
    proportions = class_counts / len(df)

    raw_allocations = proportions * n
    allocations = raw_allocations.astype(int)
    remaining = n - allocations.sum()

    if remaining > 0:
        remainders = (raw_allocations - allocations).sort_values(ascending=False)
        for label in remainders.index[:remaining]:
            allocations[label] += 1

    return allocations.to_dict()


def stratified_sample(
    df: pd.DataFrame,
    n: int,
    label_col: str = DEFAULT_LABEL_COLUMN,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Create a reproducible stratified sample of exactly n rows."""
    _validate_sampling_input(df=df, n=n, label_col=label_col)
    allocations = _allocate_sample_counts(df=df, n=n, label_col=label_col)

    sampled_parts = []
    for label, sample_count in allocations.items():
        if sample_count == 0:
            continue
        class_df = df[df[label_col] == label]
        sampled_parts.append(class_df.sample(n=sample_count, random_state=seed))

    sampled = pd.concat(sampled_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if len(sampled) != n:
        raise RuntimeError(f"Stratified sampling failed: expected {n:,} rows, got {len(sampled):,}.")

    print(f"[sample] target n={n:,}, actual n={len(sampled):,}, seed={seed}")
    distribution = sampled[label_col].value_counts().sort_index().to_frame("count")
    distribution["percentage"] = distribution["count"] / len(sampled) * 100
    print("[sample] label distribution:")
    print(distribution)

    return sampled


def stratified_train_validation_split(
    train_subset: pd.DataFrame,
    train_size: int = 90_000,
    validation_size: int = 10_000,
    label_col: str = DEFAULT_LABEL_COLUMN,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create fixed, disjoint training/validation splits from a TRAIN-only subset."""
    _validate_sampling_input(train_subset, train_size + validation_size, label_col)
    if len(train_subset) != train_size + validation_size:
        raise ValueError("The input training subset must contain exactly train_size + validation_size rows.")

    splitter = StratifiedShuffleSplit(
        n_splits=1, train_size=train_size, test_size=validation_size, random_state=seed,
    )
    train_indices, validation_indices = next(splitter.split(train_subset, train_subset[label_col]))

    train = train_subset.iloc[train_indices].reset_index(drop=True)
    validation = train_subset.iloc[validation_indices].reset_index(drop=True)

    if set(train_indices).intersection(validation_indices):
        raise RuntimeError("Training and validation subsets overlap.")

    return train, validation


def label_distribution(df: pd.DataFrame, label_col: str = DEFAULT_LABEL_COLUMN) -> dict:
    """Return fixed E/S/C/I count and percentage summaries for reporting."""
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame.")
    if len(df) == 0:
        raise ValueError("Cannot calculate a distribution for an empty DataFrame.")
    counts = df[label_col].value_counts().reindex(ESCI_LABELS, fill_value=0)
    return {
        label: {
            "count": int(counts[label]),
            "percentage": round(float(counts[label] / len(df) * 100), 6),
        }
        for label in ESCI_LABELS
    }


def subset_summary(df: pd.DataFrame, *, population_rows: int, label_col: str = DEFAULT_LABEL_COLUMN) -> dict:
    """Summarize one experimental split against its source population (for the audit report)."""
    if population_rows <= 0:
        raise ValueError("population_rows must be a positive integer.")
    return {
        "rows": len(df),
        "population_rows": population_rows,
        "percentage_of_population": round(len(df) / population_rows * 100, 6),
        "label_distribution": label_distribution(df, label_col),
    }
