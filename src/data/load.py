"""
Load and merge the raw Amazon Shopping Queries ESCI dataset.

Responsibilities:
    1. Load examples and products parquet files.
    2. Merge examples with product metadata.
    3. Filter the merged dataset by product locale + large_version.

This module does NOT:
    - clean text
    - split train/test
    - perform sampling
    - train models
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

EXAMPLES_FILE = "shopping_queries_dataset_examples.parquet"
PRODUCTS_FILE = "shopping_queries_dataset_products.parquet"

MERGE_KEYS = ["product_id", "product_locale"]


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_file_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")


def _validate_columns(df: pd.DataFrame, required_columns: list[str], dataset_name: str) -> None:
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"{dataset_name} is missing required columns: {missing_columns}")


# ---------------------------------------------------------------------
# Load raw ESCI (all locales) — kept for completeness / notebooks that
# want to inspect other locales. Not used by the US-only pipeline.
# ---------------------------------------------------------------------

def load_raw_esci(data_dir: str | Path) -> pd.DataFrame:
    """Load and merge the full raw ESCI examples + products datasets (all locales)."""
    data_dir = Path(data_dir)
    examples_path = data_dir / EXAMPLES_FILE
    products_path = data_dir / PRODUCTS_FILE

    _validate_file_exists(examples_path)
    _validate_file_exists(products_path)

    examples = pd.read_parquet(examples_path)
    products = pd.read_parquet(products_path)

    print(f"[load] examples rows: {len(examples):,}")
    print(f"[load] products rows: {len(products):,}")

    _validate_columns(examples, MERGE_KEYS, "examples")
    _validate_columns(products, MERGE_KEYS, "products")

    merged = examples.merge(
        products, on=MERGE_KEYS, how="left", suffixes=("", "_product"), validate="many_to_one",
    )
    print(f"[load] merged rows: {len(merged):,}")

    if len(merged) != len(examples):
        raise ValueError(
            "Merge changed the number of example rows. "
            "Check product_id + product_locale uniqueness in the products dataset."
        )
    return merged


# ---------------------------------------------------------------------
# US-only loader (used by the actual pipeline)
# ---------------------------------------------------------------------

def load_us_esci(data_dir: str | Path, large_version: int = 1) -> pd.DataFrame:
    """Load only Amazon ESCI large-version US rows from Parquet sources.

    The locale/version predicate is pushed down to the Parquet reader, so
    non-US row groups are never materialized in memory. This is the loader
    used for the fixed US-only experimental protocol.
    """
    data_dir = Path(data_dir)
    examples_path = data_dir / EXAMPLES_FILE
    products_path = data_dir / PRODUCTS_FILE

    _validate_file_exists(examples_path)
    _validate_file_exists(products_path)

    examples_filter = [
        ("product_locale", "==", "us"),
        ("large_version", "==", large_version),
    ]
    products_filter = [("product_locale", "==", "us")]
    examples = pd.read_parquet(examples_path, filters=examples_filter)
    products = pd.read_parquet(products_path, filters=products_filter)

    _validate_columns(examples, MERGE_KEYS, "examples")
    _validate_columns(products, MERGE_KEYS, "products")
    _validate_columns(examples, ["large_version", "split", "esci_label"], "examples")

    if examples.empty or products.empty:
        raise ValueError("No US-locale rows were found in the ESCI Parquet files.")
    if not examples["product_locale"].eq("us").all():
        raise ValueError("Examples loader returned a non-US locale row.")
    if not examples["large_version"].eq(large_version).all():
        raise ValueError("Examples loader returned a non-requested large_version row.")
    if not products["product_locale"].eq("us").all():
        raise ValueError("Products loader returned a non-US locale row.")

    merged = examples.merge(
        products, on=MERGE_KEYS, how="left", suffixes=("", "_product"), validate="many_to_one",
    )
    if len(merged) != len(examples):
        raise ValueError(
            "Merge changed the number of US example rows. "
            "Check product_id + product_locale uniqueness in the products dataset."
        )

    print(f"[load] US examples rows: {len(examples):,}")
    print(f"[load] US products rows: {len(products):,}")
    print(f"[load] US merged rows: {len(merged):,}")
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------
# Locale filtering (kept as a standalone utility for ad-hoc inspection)
# ---------------------------------------------------------------------

def filter_locale(df: pd.DataFrame, locale: str = "us") -> pd.DataFrame:
    """Filter a merged ESCI dataset down to a single product locale."""
    if "product_locale" not in df.columns:
        raise ValueError("Column 'product_locale' is required for locale filtering.")
    if not locale:
        raise ValueError("Locale must not be empty.")

    before = len(df)
    filtered = df.loc[df["product_locale"] == locale].copy().reset_index(drop=True)
    print(f"[load] filter_locale('{locale}'): {before:,} -> {len(filtered):,} rows")

    if len(filtered) == 0:
        raise ValueError(f"No rows found for product_locale='{locale}'. Check the dataset or locale value.")
    return filtered
