"""Run one fixed-split ESCI four-class classification experiment.

The official test subset (30,551 rows) is read ONLY when --final-evaluation
is explicitly supplied. Every other invocation only touches the fixed
training (90,000) and validation (10,000) splits, matching the development
protocol: test is never used for model selection, tuning, or early stopping.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import hstack

from src.data.clean import clean_dataset
from src.evaluation.metrics import ESCI_LABELS, evaluate_characteristics, evaluate_performance
from src.features.tfidf import TfidfConfig, TfidfFeatureExtractor
from src.models.lexical_svm import LinearSVMClassifier
from src.utils.config import load_config
from src.utils.seed import set_seed


def _read_split(data_dir: Path, filename: str, expected_rows: int) -> pd.DataFrame:
    path = data_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required split file not found: {path}\n"
            "Run scripts/build_us_representative_subset.py first."
        )
    frame = pd.read_parquet(path)
    if len(frame) != expected_rows:
        raise ValueError(f"{path.name} must have {expected_rows:,} rows, got {len(frame):,}.")
    if set(frame["esci_label"].dropna()) - set(ESCI_LABELS):
        raise ValueError(f"{path.name} contains unsupported ESCI labels.")
    return frame


def _assert_disjoint(left: pd.DataFrame, right: pd.DataFrame, left_name: str, right_name: str) -> None:
    if "example_id" not in left.columns or "example_id" not in right.columns:
        return
    overlap = set(left["example_id"]).intersection(right["example_id"])
    if overlap:
        raise ValueError(f"{left_name} and {right_name} overlap by {len(overlap):,} example_id values.")


def _save_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload, file, ensure_ascii=False, indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        )


def _lexical_text(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the lexical-cleaned (query, product_title) text pair for a split."""
    return frame["query_lex_clean"], frame["product_text_base_lex_clean"]


def _semantic_text(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return the semantic-cleaned (query, product_title) text pair for a split."""
    return frame["query_sem_clean"], frame["product_text_base_sem_clean"]


def _run_lexical(
    experiment: dict,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame | None,
    seed: int,
) -> tuple[dict, dict]:
    training = clean_dataset(training, arm="lexical")
    validation = clean_dataset(validation, arm="lexical")
    test = clean_dataset(test, arm="lexical") if test is not None else None

    vectorizer_config = TfidfConfig(
        max_features=experiment["max_features"], ngram_range=tuple(experiment["ngram_range"]),
    )
    train_query, train_product = _lexical_text(training)
    validation_query, validation_product = _lexical_text(validation)

    if experiment["strategy"] == "separate":
        query_vectorizer = TfidfFeatureExtractor(vectorizer_config)
        product_vectorizer = TfidfFeatureExtractor(vectorizer_config)
        # fit_transform is only ever called on the TRAINING text.
        X_train = hstack([
            query_vectorizer.fit_transform(train_query),
            product_vectorizer.fit_transform(train_product),
        ]).tocsr()
        X_validation = hstack([
            query_vectorizer.transform(validation_query),
            product_vectorizer.transform(validation_product),
        ]).tocsr()
        artifact = {"query_vectorizer": query_vectorizer, "product_vectorizer": product_vectorizer}
        if test is not None:
            test_query, test_product = _lexical_text(test)
            X_test = hstack([
                query_vectorizer.transform(test_query),
                product_vectorizer.transform(test_product),
            ]).tocsr()

    elif experiment["strategy"] == "concatenated":
        vectorizer = TfidfFeatureExtractor(vectorizer_config)
        X_train = vectorizer.fit_transform(train_query + " [SEP] " + train_product)
        X_validation = vectorizer.transform(validation_query + " [SEP] " + validation_product)
        artifact = {"vectorizer": vectorizer}
        if test is not None:
            test_query, test_product = _lexical_text(test)
            X_test = vectorizer.transform(test_query + " [SEP] " + test_product)

    else:
        raise ValueError(f"Unknown lexical strategy: {experiment['strategy']}")

    model = LinearSVMClassifier(C=experiment["C"], random_state=seed).fit(X_train, training["esci_label"])
    validation_predictions = model.predict(X_validation)
    results = {
        "validation": {
            **evaluate_performance(validation["esci_label"], validation_predictions),
            **evaluate_characteristics(validation["esci_label"], validation_predictions, list(ESCI_LABELS)),
        }
    }
    if test is not None:
        test_predictions = model.predict(X_test)
        results["final_test"] = {
            **evaluate_performance(test["esci_label"], test_predictions),
            **evaluate_characteristics(test["esci_label"], test_predictions, list(ESCI_LABELS)),
        }

    artifact["model"] = model
    return artifact, results


def _run_semantic(
    experiment: dict,
    training: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame | None,
    output_dir: Path,
    seed: int,
    resume: bool,
) -> dict:
    # Imported lazily so the lexical arm never requires torch/transformers
    # to be installed (keeps CPU-only / no-GPU development environments usable).
    from src.models.semantic_transformer import TransformerClassifier

    training = clean_dataset(training, arm="semantic")
    validation = clean_dataset(validation, arm="semantic")

    train_query, train_product = _semantic_text(training)
    validation_query, validation_product = _semantic_text(validation)

    model = TransformerClassifier(
        checkpoint=experiment["checkpoint"],
        max_sequence_length=experiment["max_sequence_length"],
        learning_rate=experiment["learning_rate"],
        batch_size=experiment["batch_size"],
        epochs=experiment["max_epochs"],
        seed=seed,
        early_stopping_patience=experiment["early_stopping_patience"],
        checkpoint_save_steps=experiment["checkpoint_save_steps"],
        progress_report_steps=experiment["progress_report_steps"],
    )
    model.fit(
        train_query, train_product, training["esci_label"],
        output_dir=str(output_dir),
        validation_queries=validation_query,
        validation_product_texts=validation_product,
        validation_labels=validation["esci_label"],
        resume_from_checkpoint=resume,
    )

    validation_predictions = model.predict(validation_query, validation_product)
    results = {
        "validation": {
            **evaluate_performance(validation["esci_label"], validation_predictions),
            **evaluate_characteristics(validation["esci_label"], validation_predictions, list(ESCI_LABELS)),
        },
        "loss_history": model.get_loss_history(),
    }

    if test is not None:
        test = clean_dataset(test, arm="semantic")
        test_query, test_product = _semantic_text(test)
        test_predictions = model.predict(test_query, test_product)
        results["final_test"] = {
            **evaluate_performance(test["esci_label"], test_predictions),
            **evaluate_characteristics(test["esci_label"], test_predictions, list(ESCI_LABELS)),
        }

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/main_experiment.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--final-evaluation", action="store_true")
    parser.add_argument(
        "--restart-semantic-training",
        action="store_true",
        help="Ignore saved transformer checkpoints and begin semantic training again.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.experiment not in config["experiments"]:
        raise ValueError(
            f"Unknown experiment: {args.experiment}. Available: {list(config['experiments'])}"
        )

    experiment = config["experiments"][args.experiment]
    data_config = config["data"]
    output_dir = Path(args.output_dir) / args.experiment
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(data_config["seed"])

    training = _read_split(Path(args.data_dir), config["splits"]["training_file"], data_config["training_rows"])
    validation = _read_split(Path(args.data_dir), config["splits"]["validation_file"], data_config["validation_rows"])
    _assert_disjoint(training, validation, "training", "validation")

    test = None
    if args.final_evaluation:
        test = _read_split(Path(args.data_dir), config["splits"]["test_file"], data_config["test_subset_rows"])
        _assert_disjoint(training, test, "training", "test")
        _assert_disjoint(validation, test, "validation", "test")

    if experiment["arm"] == "lexical":
        artifact, results = _run_lexical(experiment, training, validation, test, data_config["seed"])
        joblib.dump(artifact, output_dir / "artifact.joblib")
    elif experiment["arm"] == "semantic":
        results = _run_semantic(
            experiment, training, validation, test, output_dir, data_config["seed"],
            resume=not args.restart_semantic_training,
        )
    else:
        raise ValueError(f"Unsupported arm: {experiment['arm']}")

    results.update({"experiment": args.experiment, "final_evaluation_ran": args.final_evaluation})
    _save_json(results, output_dir / "results.json")
    print(f"[output] Results written to: {output_dir / 'results.json'}")
    for split_name in ("validation", "final_test"):
        if split_name in results:
            summary = {
                metric: results[split_name][metric]
                for metric in ("macro_f1", "weighted_f1", "accuracy", "micro_f1")
            }
            print(f"[{split_name}] {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
