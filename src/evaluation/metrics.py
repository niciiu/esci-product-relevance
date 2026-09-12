"""Metrics for four-class ESCI relevance classification."""
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

ESCI_LABELS = ("E", "S", "C", "I")


def evaluate_performance(y_true, y_pred) -> dict:
    """Primary + secondary metrics per the evaluation protocol (macro/weighted F1, accuracy, micro F1)."""
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=ESCI_LABELS, zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", labels=ESCI_LABELS, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "micro_f1": f1_score(y_true, y_pred, average="micro", labels=ESCI_LABELS, zero_division=0),
    }


def evaluate_characteristics(y_true, y_pred, labels: list[str]) -> dict:
    """Per-class F1, full classification report, and confusion matrix for error analysis."""
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=labels, output_dict=True, zero_division=0,
    )
    return {
        "per_class_f1": {label: report[label]["f1-score"] for label in labels},
        "per_class_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
