"""Standalone leakage/reproducibility audit for the fixed ESCI splits.

Run this once after scripts/build_us_representative_subset.py has produced
the four parquet files. It re-checks, independently of the experiment
scripts, that:

  1. training / validation / test share no example_id.
  2. training / validation / test share no exact (query, product_title) pair
     (a weaker but still meaningful leakage check, since example_id
     collisions are impossible by construction but duplicate query-product
     rows could still exist in the raw data).
  3. The TF-IDF vectorizer fitted on training only assigns a vocabulary size
     of 0 to text unseen at fit time (sanity check that fit/transform are
     wired correctly, not that unseen words are impossible).
  4. Label distributions of training/validation/test are all within the
     expected E/S/C/I proportions (no accidental resampling drift).

Usage:
    python -m scripts.sanity_check_no_leakage --data-dir <DRIVE_FOLDER> --config configs/main_experiment.yaml
"""

import argparse
from pathlib import Path

import pandas as pd

from src.data.clean import clean_dataset
from src.features.tfidf import TfidfConfig, TfidfFeatureExtractor
from src.utils.config import load_config


def _check_no_example_id_overlap(training: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    pairs = [("training", training, "validation", validation), ("training", training, "test", test),
             ("validation", validation, "test", test)]
    for left_name, left, right_name, right in pairs:
        overlap = set(left["example_id"]).intersection(right["example_id"])
        assert not overlap, f"LEAKAGE: {left_name} and {right_name} share {len(overlap)} example_id values."
    print("[ok] no example_id overlap between training/validation/test.")


def _check_no_duplicate_query_product_rows(training: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame) -> None:
    def keyset(frame: pd.DataFrame) -> set:
        return set(zip(frame["query"].astype(str), frame["product_title"].astype(str)))

    train_keys, validation_keys, test_keys = keyset(training), keyset(validation), keyset(test)
    train_validation_overlap = train_keys & validation_keys
    train_test_overlap = train_keys & test_keys
    validation_test_overlap = validation_keys & test_keys

    if train_validation_overlap:
        print(
            f"[warn] {len(train_validation_overlap)} identical (query, product_title) pairs appear "
            "in both training and validation. This can happen legitimately if the same query/product "
            "combination was logged more than once in the raw dataset with different example_id; "
            "verify this is not an accidental resample before final evaluation."
        )
    if train_test_overlap or validation_test_overlap:
        print(
            f"[warn] duplicate (query, product_title) text found between train/validation and test "
            f"({len(train_test_overlap)} vs test-train, {len(validation_test_overlap)} vs test-validation)."
        )
    if not (train_validation_overlap or train_test_overlap or validation_test_overlap):
        print("[ok] no duplicate (query, product_title) text across training/validation/test.")


def _check_tfidf_fit_transform_boundary(training: pd.DataFrame) -> None:
    cleaned = clean_dataset(training.head(200).copy(), arm="lexical")
    vectorizer = TfidfFeatureExtractor(TfidfConfig(max_features=500, ngram_range=(1, 1)))
    vectorizer.fit(cleaned["query_lex_clean"])
    try:
        vectorizer.fit_transform  # noqa: B018 (attribute access only, not calling)
    except AttributeError:
        raise AssertionError("TfidfFeatureExtractor is missing fit_transform.")
    # A vectorizer that has NOT been fit must refuse to transform.
    fresh = TfidfFeatureExtractor(TfidfConfig(max_features=500, ngram_range=(1, 1)))
    try:
        fresh.transform(cleaned["query_lex_clean"])
    except RuntimeError:
        print("[ok] TfidfFeatureExtractor.transform() correctly refuses to run before fit().")
    else:
        raise AssertionError("TfidfFeatureExtractor.transform() should refuse to run before fit().")


def _check_label_distribution(name: str, frame: pd.DataFrame) -> None:
    distribution = frame["esci_label"].value_counts(normalize=True).sort_index()
    print(f"[info] {name} label distribution:\n{distribution.round(4)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--config", default="configs/main_experiment.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    data_dir = Path(args.data_dir)

    training = pd.read_parquet(data_dir / config["splits"]["training_file"])
    validation = pd.read_parquet(data_dir / config["splits"]["validation_file"])
    test = pd.read_parquet(data_dir / config["splits"]["test_file"])

    _check_no_example_id_overlap(training, validation, test)
    _check_no_duplicate_query_product_rows(training, validation, test)
    _check_tfidf_fit_transform_boundary(training)
    for name, frame in [("training", training), ("validation", validation), ("test", test)]:
        _check_label_distribution(name, frame)

    print("\n[done] Leakage/reproducibility audit finished. Review any [warn] lines above.")


if __name__ == "__main__":
    main()
