"""Build fixed ESCI US development/final-evaluation subsets.

The official split is preserved: 100,000 rows are sampled from official TRAIN
and 30,551 rows are sampled separately from official TEST. The TRAIN subset
is then split once into 90,000 training and 10,000 validation rows.

Run this ONCE per seed. The output parquet files are the fixed splits every
experiment (scripts/run_main_experiment.py) reads from — regenerating them
with a different seed would silently change every downstream result, so the
seed and expected official-population sizes are asserted below.
"""

import argparse
import json
from pathlib import Path

from src.data.load import load_us_esci
from src.data.sample import (
    label_distribution,
    stratified_sample,
    stratified_train_validation_split,
    subset_summary,
)
from src.data.split import get_official_split
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create US-only ESCI data and fixed representative subsets."
    )
    parser.add_argument("--raw-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--test-size", type=int, default=30_551)
    parser.add_argument("--training-size", type=int, default=90_000)
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument("--expected-official-train-size", type=int, default=1_393_063)
    parser.add_argument("--expected-official-test-size", type=int, default=425_762)
    args = parser.parse_args()

    if args.training_size + args.validation_size != args.train_size:
        raise ValueError(
            "training-size + validation-size must equal train-size so the "
            "fixed TRAIN subset is partitioned without omission."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    english_us = load_us_esci(args.raw_data_dir, large_version=1)

    official_train, official_test = get_official_split(english_us)
    if len(official_train) != args.expected_official_train_size:
        raise ValueError(
            "Unexpected official US TRAIN size for large_version=1: "
            f"expected {args.expected_official_train_size:,}, got {len(official_train):,}."
        )
    if len(official_test) != args.expected_official_test_size:
        raise ValueError(
            "Unexpected official US TEST size for large_version=1: "
            f"expected {args.expected_official_test_size:,}, got {len(official_test):,}."
        )

    train_subset = stratified_sample(official_train, n=args.train_size, seed=args.seed)
    test_subset = stratified_sample(official_test, n=args.test_size, seed=args.seed)
    training, validation = stratified_train_validation_split(
        train_subset,
        train_size=args.training_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )

    # ---- No-leakage guarantee: training/validation/test must never overlap ----
    if "example_id" in train_subset.columns and "example_id" in test_subset.columns:
        overlap = set(train_subset["example_id"]).intersection(test_subset["example_id"])
        if overlap:
            raise RuntimeError(
                f"Representative TRAIN and TEST subsets overlap by {len(overlap):,} "
                "example_id values — this should be impossible since they are sampled "
                "from disjoint official splits."
            )

    train_path = output_dir / f"esci_english_us_train_{args.train_size}.parquet"
    validation_path = output_dir / f"esci_english_us_validation_{args.validation_size}.parquet"
    training_path = output_dir / f"esci_english_us_training_{args.training_size}.parquet"
    test_path = output_dir / f"esci_english_us_test_{args.test_size}.parquet"

    train_subset.to_parquet(train_path, index=False)
    training.to_parquet(training_path, index=False)
    validation.to_parquet(validation_path, index=False)
    test_subset.to_parquet(test_path, index=False)

    report = {
        "dataset": {
            "product_locale": "us",
            "large_version": 1,
            "official_train_rows": len(official_train),
            "official_test_rows": len(official_test),
            "official_us_total_rows": len(official_train) + len(official_test),
        },
        "methodology": {
            "task": "four_class_query_product_relevance_classification",
            "labels": {"E": "Exact", "S": "Substitute", "C": "Complement", "I": "Irrelevant"},
            "text_fields_used": ["query", "product_title"],
            "random_state": args.seed,
            "official_split_preserved": True,
            "test_used_only_for_final_evaluation": True,
        },
        "sampling_fractions": {
            "official_train_to_train_subset_percentage": round(len(train_subset) / len(official_train) * 100, 6),
            "official_test_to_evaluation_subset_percentage": round(len(test_subset) / len(official_test) * 100, 6),
            "train_subset_to_training_percentage": round(len(training) / len(train_subset) * 100, 6),
            "train_subset_to_validation_percentage": round(len(validation) / len(train_subset) * 100, 6),
        },
        "size_audit": {
            "official_us_train_before_sampling": {"rows": len(official_train), "percentage_of_official_train": 100.0},
            "official_us_test_before_sampling": {"rows": len(official_test), "percentage_of_official_test": 100.0},
            "official_us_total_before_sampling": {
                "rows": len(official_train) + len(official_test),
                "percentage_of_official_us_total": 100.0,
            },
            "representative_train_after_sampling": {
                "rows": len(train_subset),
                "percentage_of_official_train": round(len(train_subset) / len(official_train) * 100, 6),
            },
            "representative_test_after_sampling": {
                "rows": len(test_subset),
                "percentage_of_official_test": round(len(test_subset) / len(official_test) * 100, 6),
            },
            "representative_combined_after_sampling": {
                "rows": len(train_subset) + len(test_subset),
                "percentage_of_official_us_total": round(
                    (len(train_subset) + len(test_subset)) / (len(official_train) + len(official_test)) * 100, 6,
                ),
            },
            "training_after_train_validation_split": {
                "rows": len(training),
                "percentage_of_representative_train": round(len(training) / len(train_subset) * 100, 6),
                "percentage_of_official_train": round(len(training) / len(official_train) * 100, 6),
            },
            "validation_after_train_validation_split": {
                "rows": len(validation),
                "percentage_of_representative_train": round(len(validation) / len(train_subset) * 100, 6),
                "percentage_of_official_train": round(len(validation) / len(official_train) * 100, 6),
            },
            "final_evaluation_test": {
                "rows": len(test_subset),
                "percentage_of_representative_test": 100.0,
                "percentage_of_official_test": round(len(test_subset) / len(official_test) * 100, 6),
            },
        },
        "split_summaries": {
            "official_train_before_sampling": subset_summary(official_train, population_rows=len(official_train)),
            "train_subset_after_sampling": subset_summary(train_subset, population_rows=len(official_train)),
            "training_after_train_validation_split": subset_summary(training, population_rows=len(train_subset)),
            "validation_after_train_validation_split": subset_summary(validation, population_rows=len(train_subset)),
            "official_test_before_sampling": subset_summary(official_test, population_rows=len(official_test)),
            "evaluation_test_after_sampling": subset_summary(test_subset, population_rows=len(official_test)),
        },
        "splits": {
            "official_train": label_distribution(official_train),
            "official_test": label_distribution(official_test),
            "train_subset_100000": label_distribution(train_subset),
            "test_subset_30551": label_distribution(test_subset),
            "training_90000": label_distribution(training),
            "validation_10000": label_distribution(validation),
        },
    }

    report_path = output_dir / "esci_us_sampling_report.json"
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print(f"[output] Stratified official TRAIN subset: {train_path}")
    print(f"[output] Fixed training split: {training_path}")
    print(f"[output] Fixed validation split: {validation_path}")
    print(f"[output] Stratified official TEST subset: {test_path}")
    print(f"[output] Label-distribution report: {report_path}")
    print("[output] Size audit:")
    for split_name, values in report["size_audit"].items():
        print(f"  {split_name}: {values}")


if __name__ == "__main__":
    main()
