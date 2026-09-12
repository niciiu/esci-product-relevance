"""
Official ESCI train/test split utilities.

The dataset's original `split` column is used as-is. No manual train/test
splitting is performed here — that would risk leaking rows across the
official boundary. The official test set must only be used for final
evaluation (see scripts/run_main_experiment.py --final-evaluation).
"""

from pathlib import Path

import pandas as pd

SPLIT_COLUMN = "split"
TRAIN_VALUE = "train"
TEST_VALUE = "test"

LOCKED_TEST_FILENAME = "test_official_locked.parquet"


def _validate_split_column(df: pd.DataFrame) -> None:
    if SPLIT_COLUMN not in df.columns:
        raise ValueError(f"Required column '{SPLIT_COLUMN}' is not present in the DataFrame.")


def _validate_split_values(df: pd.DataFrame) -> None:
    split_values = set(df[SPLIT_COLUMN].dropna().unique())
    unexpected = split_values - {TRAIN_VALUE, TEST_VALUE}
    if unexpected:
        raise ValueError(f"Unexpected values found in '{SPLIT_COLUMN}': {unexpected}")
    if df[SPLIT_COLUMN].isna().any():
        raise ValueError(f"Column '{SPLIT_COLUMN}' contains missing values.")


def get_official_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate ESCI data using the dataset's official `split` column.

    Returns
    -------
    (train, test) : both are official, disjoint-by-construction subsets.
    """
    _validate_split_column(df)
    _validate_split_values(df)

    train = df.loc[df[SPLIT_COLUMN] == TRAIN_VALUE].copy().reset_index(drop=True)
    test = df.loc[df[SPLIT_COLUMN] == TEST_VALUE].copy().reset_index(drop=True)

    if len(train) == 0:
        raise ValueError("Official train split is empty.")
    if len(test) == 0:
        raise ValueError("Official test split is empty.")

    print(f"[split] official train: {len(train):,} rows")
    print(f"[split] official test: {len(test):,} rows")
    return train, test


def lock_test_set(test_df: pd.DataFrame, output_dir: str | Path) -> Path:
    """Save the official test set as a locked parquet file.

    Refuses to overwrite an existing locked test file, so the final
    evaluation set can never be silently regenerated with a different seed.
    """
    if not isinstance(test_df, pd.DataFrame):
        raise TypeError("test_df must be a pandas DataFrame.")
    if len(test_df) == 0:
        raise ValueError("Cannot lock an empty test set.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / LOCKED_TEST_FILENAME

    if output_path.exists():
        raise FileExistsError(
            f"{output_path} already exists. The official test set is locked "
            "and must not be overwritten automatically."
        )

    test_df.to_parquet(output_path, index=False)
    print(f"[split] official test locked at: {output_path}")
    return output_path
