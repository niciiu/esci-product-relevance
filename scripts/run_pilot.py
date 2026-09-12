"""Run a small, reproducible semantic (DeBERTa-v3-base) development pilot.

Purpose: sanity-check the training loop (tokenization, checkpointing,
early stopping, resume-from-checkpoint) on a few thousand rows before
committing GPU time to the full 90K/10K run. The pilot samples only the
official TRAIN split, then creates disjoint training/validation
partitions — the official TEST split is never read here.
"""

import argparse
import json
from pathlib import Path

from src.data.clean import clean_dataset
from src.data.load import load_us_esci
from src.data.sample import stratified_sample, stratified_train_validation_split
from src.data.split import get_official_split
from src.evaluation.metrics import ESCI_LABELS, evaluate_characteristics, evaluate_performance
from src.models.semantic_transformer import TransformerClassifier
from src.utils.config import load_config
from src.utils.seed import set_seed


def _save_json(payload: dict, path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload, file, ensure_ascii=False, indent=2,
            default=lambda value: value.item() if hasattr(value, "item") else str(value),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", "--data_dir", dest="data_dir", required=True)
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", required=True)
    parser.add_argument(
        "--restart-semantic-training", action="store_true",
        help="Ignore an existing pilot checkpoint and start again.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_config, model_config = config["data"], config["model"]
    sample_size = data_config["sample_size"]
    training_size = data_config["training_size"]
    validation_size = data_config["validation_size"]

    if training_size + validation_size != sample_size:
        raise ValueError("training_size + validation_size must equal sample_size.")
    if config["arm"] != "semantic" or model_config["type"] != "transformer":
        raise ValueError("run_pilot currently supports semantic transformer pilots only.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(data_config["seed"])

    english_us = load_us_esci(args.data_dir)
    official_train, _official_test = get_official_split(english_us)

    pilot_sample = stratified_sample(official_train, n=sample_size, seed=data_config["seed"])
    training, validation = stratified_train_validation_split(
        pilot_sample, train_size=training_size, validation_size=validation_size, seed=data_config["seed"],
    )
    training = clean_dataset(training, arm="semantic")
    validation = clean_dataset(validation, arm="semantic")

    model = TransformerClassifier(
        checkpoint=model_config["checkpoint"],
        max_sequence_length=model_config["max_sequence_length"],
        learning_rate=model_config["learning_rate"],
        batch_size=model_config["batch_size"],
        epochs=model_config["epochs"],
        seed=data_config["seed"],
        early_stopping_patience=model_config["early_stopping_patience"],
        checkpoint_save_steps=model_config["checkpoint_save_steps"],
        progress_report_steps=model_config.get("progress_report_steps", 50),
    )
    model.fit(
        training["query_sem_clean"], training["product_text_base_sem_clean"], training["esci_label"],
        output_dir=str(output_dir),
        validation_queries=validation["query_sem_clean"],
        validation_product_texts=validation["product_text_base_sem_clean"],
        validation_labels=validation["esci_label"],
        resume_from_checkpoint=not args.restart_semantic_training,
    )
    predictions = model.predict(validation["query_sem_clean"], validation["product_text_base_sem_clean"])

    results = {
        "experiment": config["experiment_name"],
        "final_evaluation_ran": False,
        "note": "Validation-only pilot; the official TEST split was not used.",
        "data": {
            "pilot_sample_rows": len(pilot_sample),
            "training_rows": len(training),
            "validation_rows": len(validation),
            "seed": data_config["seed"],
        },
        "validation": {
            **evaluate_performance(validation["esci_label"], predictions),
            **evaluate_characteristics(validation["esci_label"], predictions, list(ESCI_LABELS)),
        },
        "loss_history": model.get_loss_history(),
    }
    _save_json(results, output_dir / "results.json")
    print(json.dumps({k: v for k, v in results.items() if k != "loss_history"}, indent=2, default=str))


if __name__ == "__main__":
    main()
