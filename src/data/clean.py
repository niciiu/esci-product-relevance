"""
Data cleaning pipeline for ESCI.

SCOPE (locked): only `query` and `product_title` are ever read or cleaned.
`product_description` and `product_bullet_point` are intentionally never
touched, even if present in the raw dataset — the study models
query-product_title relevance only.

Pipeline:

    Raw data (query, product_title, esci_label)
        |
    Official train/test split   (src/data/split.py)
        |
    Base cleaning (product_title only)
        |
    +------------------+------------------+
    |                                     |
    Lexical branch                  Semantic branch
    |                                     |
    TF-IDF                           Transformer (DeBERTa-v3-base)

Notes:
    - This module does NOT perform train/test splitting.
    - Deduplication is intentionally not performed automatically; if the
      thesis later needs it, it must be applied to the training split only,
      never to validation or test.
"""

import html
import re

import pandas as pd

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

# Locked scope: product_title is the ONLY product-side text field used.
TEXT_COLUMNS = ["product_title"]

BASE_CLEAN_SUFFIX = "_base_clean"
LEXICAL_CLEAN_SUFFIX = "_lex_clean"
SEMANTIC_CLEAN_SUFFIX = "_sem_clean"

PRODUCT_TEXT_BASE_COLUMN = "product_text_base"

REQUIRED_RAW_COLUMNS = ["query", "esci_label", *TEXT_COLUMNS]


# ---------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------

def _validate_text_columns(df: pd.DataFrame) -> None:
    missing_columns = [col for col in TEXT_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required text columns: {missing_columns}")


def _validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = [col for col in REQUIRED_RAW_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Missing required raw columns for the locked query+product_title+label "
            f"scope: {missing_columns}"
        )


# ---------------------------------------------------------------------
# Base cleaning
# ---------------------------------------------------------------------

def fill_missing_text(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing product titles (and query) with empty strings."""
    _validate_text_columns(df)
    df = df.copy()

    for col in [*TEXT_COLUMNS, "query"]:
        missing_pct = df[col].isna().mean() * 100
        print(f"[clean] {col} missing: {missing_pct:.2f}%")
        df[col] = df[col].fillna("")

    return df


def _strip_html_one(text: str) -> str:
    """Remove HTML entities, HTML tags, control characters, extra whitespace."""
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_html_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Apply shared structural cleaning to product_title. Stored with '_base_clean' suffix."""
    _validate_text_columns(df)
    df = df.copy()

    for col in TEXT_COLUMNS:
        df[f"{col}{BASE_CLEAN_SUFFIX}"] = df[col].apply(_strip_html_one)

    return df


def build_product_text_base(df: pd.DataFrame) -> pd.DataFrame:
    """Build the model's product text directly from the cleaned product_title.

    Because TEXT_COLUMNS contains only product_title, this is effectively an
    alias, but is kept as a separate, explicitly named column so downstream
    code (lexical/semantic branches) never has to know the source column name.
    """
    base_columns = [f"{col}{BASE_CLEAN_SUFFIX}" for col in TEXT_COLUMNS]
    missing_columns = [col for col in base_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing base-cleaned columns: {missing_columns}")

    df = df.copy()
    df[PRODUCT_TEXT_BASE_COLUMN] = df[base_columns[0]]
    return df


def base_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run the shared base cleaning pipeline: fill -> strip HTML/noise -> build product text."""
    _validate_required_columns(df)
    df = df.copy()
    df = fill_missing_text(df)
    df = clean_html_noise(df)
    df = build_product_text_base(df)
    return df


# ---------------------------------------------------------------------
# Lexical cleaning
# ---------------------------------------------------------------------

def _lexical_normalize_one(text: str) -> str:
    """Lowercase, strip punctuation/symbols, normalize whitespace. No stopword removal."""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_lexical(df: pd.DataFrame) -> pd.DataFrame:
    """Create lexical-cleaned versions of query and product_text_base."""
    df = df.copy()
    source_columns = ["query", PRODUCT_TEXT_BASE_COLUMN]
    for col in source_columns:
        df[f"{col}{LEXICAL_CLEAN_SUFFIX}"] = df[col].apply(_lexical_normalize_one)
    return df


# ---------------------------------------------------------------------
# Semantic cleaning
# ---------------------------------------------------------------------

def _semantic_normalize_one(text: str) -> str:
    """Whitespace-only normalization. Casing/punctuation preserved for the transformer tokenizer."""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_for_semantic(df: pd.DataFrame) -> pd.DataFrame:
    """Create semantic-cleaned versions of query and product_text_base."""
    df = df.copy()
    source_columns = ["query", PRODUCT_TEXT_BASE_COLUMN]
    for col in source_columns:
        df[f"{col}{SEMANTIC_CLEAN_SUFFIX}"] = df[col].apply(_semantic_normalize_one)
    return df


# ---------------------------------------------------------------------
# Full cleaning pipeline
# ---------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame, arm: str = "both") -> pd.DataFrame:
    """Run the ESCI cleaning pipeline for a single split.

    Parameters
    ----------
    df : Input ESCI DataFrame containing at least query, product_title, esci_label.
    arm : {"lexical", "semantic", "both"} — which branch(es) to produce.

    Notes
    -----
    This function must be called separately, per split (training/validation/
    test) — never on data that mixes splits, and text cleaning here is a pure
    per-row function (no cross-row statistics), so calling it independently
    per split introduces no leakage.
    """
    if arm not in {"lexical", "semantic", "both"}:
        raise ValueError(f"Unknown cleaning arm: {arm}. Expected 'lexical', 'semantic', or 'both'.")

    before = len(df)
    df = base_clean(df)

    if arm in {"lexical", "both"}:
        df = clean_for_lexical(df)
    if arm in {"semantic", "both"}:
        df = clean_for_semantic(df)

    print(f"[clean] total pipeline ({arm}): {before} -> {len(df)} rows")
    return df
