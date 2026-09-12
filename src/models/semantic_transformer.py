"""Hugging Face transformer sequence classifier for ESCI relevance labels.

Locked to a single semantic backbone for this thesis: microsoft/deberta-v3-base
(configured in configs/main_experiment.yaml). BERT-base and RoBERTa-base were
evaluated during early development but were dropped per pembimbing's feedback,
so this module is intentionally checkpoint-agnostic (it will happily load any
AutoModelForSequenceClassification checkpoint) rather than hard-coding DeBERTa,
in case a future revision needs to swap it back in for an ablation.

This class only ever sees already-cleaned text produced by
`src.data.clean.clean_dataset(arm="semantic")`. It does not read files,
split data, or know anything about train/validation/test boundaries — the
caller (scripts/run_main_experiment.py) is responsible for that separation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

LABEL_TO_ID = {"E": 0, "S": 1, "C": 2, "I": 3}
ID_TO_LABEL = {label_id: label for label, label_id in LABEL_TO_ID.items()}


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class _PairedTextDataset(Dataset):
    """Lazily tokenizes (query, product_title) pairs for the HF Trainer."""

    def __init__(
        self,
        tokenizer,
        queries: Sequence[str],
        product_texts: Sequence[str],
        max_sequence_length: int,
        labels: Sequence[str] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._queries = list(queries)
        self._product_texts = list(product_texts)
        self._max_sequence_length = max_sequence_length
        self._labels = list(labels) if labels is not None else None

        if len(self._queries) != len(self._product_texts):
            raise ValueError("queries and product_texts must have the same length.")
        if self._labels is not None and len(self._labels) != len(self._queries):
            raise ValueError("labels must match the number of query/product pairs.")
        if self._labels is not None:
            unknown = set(self._labels) - set(LABEL_TO_ID)
            if unknown:
                raise ValueError(f"Unsupported ESCI labels in dataset: {sorted(unknown)}")

    def __len__(self) -> int:
        return len(self._queries)

    def __getitem__(self, index: int) -> dict:
        encoding = self._tokenizer(
            str(self._queries[index]),
            str(self._product_texts[index]),
            truncation=True,
            max_length=self._max_sequence_length,
            padding="max_length",
        )
        item = {key: torch.tensor(value) for key, value in encoding.items()}
        if self._labels is not None:
            item["labels"] = torch.tensor(LABEL_TO_ID[self._labels[index]], dtype=torch.long)
        return item


# ---------------------------------------------------------------------
# Trainer helpers
# ---------------------------------------------------------------------

def _compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(labels, predictions, average="weighted", zero_division=0),
    }


class _LossHistoryCallback(TrainerCallback):
    """Collects every logged training/eval scalar so it can be persisted to results.json."""

    def __init__(self) -> None:
        self.history: list[dict] = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401, ANN001
        if not logs:
            return
        entry = {"step": state.global_step, "epoch": state.epoch}
        entry.update({key: value for key, value in logs.items() if isinstance(value, (int, float))})
        self.history.append(entry)


def _find_resume_checkpoint(output_dir: Path) -> str | None:
    """Return the most recent *complete* checkpoint directory, or None.

    A checkpoint is considered complete only if it has a trainer_state.json
    (written last by the Trainer), which guards against resuming from a
    checkpoint that was interrupted mid-write (e.g. Colab disconnect).
    """
    if not output_dir.exists():
        return None

    checkpoints = []
    for candidate in output_dir.glob("checkpoint-*"):
        try:
            step = int(candidate.name.split("-")[-1])
        except ValueError:
            continue
        if (candidate / "trainer_state.json").exists():
            checkpoints.append((step, candidate))

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda pair: pair[0])
    return str(checkpoints[-1][1])


# ---------------------------------------------------------------------
# Public classifier
# ---------------------------------------------------------------------

class TransformerClassifier:
    """Fine-tunes a Hugging Face sequence-classification checkpoint for ESCI labels.

    Model selection uses validation loss only (never test), matching the
    thesis's development protocol: the official test split is read exclusively
    by scripts/run_main_experiment.py --final-evaluation, after all
    architecture/hyperparameter decisions are locked.
    """

    def __init__(
        self,
        checkpoint: str,
        max_sequence_length: int = 128,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        epochs: int = 10,
        seed: int = 42,
        early_stopping_patience: int = 2,
        checkpoint_save_steps: int = 500,
        progress_report_steps: int = 50,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.06,
    ) -> None:
        self.checkpoint = checkpoint
        self.max_sequence_length = max_sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience
        self.checkpoint_save_steps = checkpoint_save_steps
        self.progress_report_steps = progress_report_steps
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            num_labels=len(LABEL_TO_ID),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )
        self._loss_history_callback = _LossHistoryCallback()
        self._is_fitted = False

    # -------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------

    def fit(
        self,
        train_queries: Sequence[str],
        train_product_texts: Sequence[str],
        train_labels: Sequence[str],
        output_dir: str,
        validation_queries: Sequence[str],
        validation_product_texts: Sequence[str],
        validation_labels: Sequence[str],
        resume_from_checkpoint: bool = True,
    ) -> "TransformerClassifier":
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        train_dataset = _PairedTextDataset(
            self.tokenizer, train_queries, train_product_texts, self.max_sequence_length, train_labels,
        )
        validation_dataset = _PairedTextDataset(
            self.tokenizer, validation_queries, validation_product_texts, self.max_sequence_length, validation_labels,
        )

        resume_checkpoint = _find_resume_checkpoint(output_path) if resume_from_checkpoint else None
        if resume_checkpoint:
            print(f"[semantic] resuming training from checkpoint: {resume_checkpoint}")
        elif resume_from_checkpoint:
            print("[semantic] no complete checkpoint found; starting training from scratch.")

        training_args = TrainingArguments(
            output_dir=str(output_path),
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=max(self.batch_size * 2, self.batch_size),
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            evaluation_strategy="steps",
            eval_steps=self.checkpoint_save_steps,
            save_strategy="steps",
            save_steps=self.checkpoint_save_steps,
            save_total_limit=3,
            logging_steps=self.progress_report_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=self.seed,
            data_seed=self.seed,
            report_to=[],
            disable_tqdm=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            compute_metrics=_compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience),
                self._loss_history_callback,
            ],
        )

        trainer.train(resume_from_checkpoint=resume_checkpoint)
        self.model = trainer.model
        self._is_fitted = True

        best_model_dir = output_path / "best_model"
        trainer.save_model(str(best_model_dir))
        self.tokenizer.save_pretrained(str(best_model_dir))
        print(f"[semantic] best checkpoint (by validation loss) saved to: {best_model_dir}")

        return self

    # -------------------------------------------------------------
    # Predict
    # -------------------------------------------------------------

    def predict(self, queries: Sequence[str], product_texts: Sequence[str]) -> list[str]:
        self._require_fitted()
        dataset = _PairedTextDataset(self.tokenizer, queries, product_texts, self.max_sequence_length)

        device = next(self.model.parameters()).device
        self.model.eval()

        eval_batch_size = max(self.batch_size * 2, self.batch_size)
        predictions: list[int] = []

        with torch.no_grad():
            for start in range(0, len(dataset), eval_batch_size):
                end = min(start + eval_batch_size, len(dataset))
                batch_items = [dataset[i] for i in range(start, end)]
                batch = {
                    key: torch.stack([item[key] for item in batch_items]).to(device)
                    for key in batch_items[0]
                }
                logits = self.model(**batch).logits
                predictions.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        return [ID_TO_LABEL[prediction] for prediction in predictions]

    # -------------------------------------------------------------
    # Introspection
    # -------------------------------------------------------------

    def get_loss_history(self) -> list[dict]:
        return self._loss_history_callback.history

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("TransformerClassifier has not been fitted. Call fit() first.")
