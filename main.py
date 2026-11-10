from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from pyTsetlinMachine.tm import MultiClassTsetlinMachine


SEED = 42
DEFAULT_DATASET = Path(__file__).parent / "Dataset" / "20_newsgroups"
DEFAULT_RESULTS = Path(__file__).parent / "results"


def load_local_newsgroups(dataset_dir: Path) -> tuple[list[str], np.ndarray]:
    """Load text files whose parent directory is the document category."""
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    documents: list[str] = []
    labels: list[str] = []
    category_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if len(category_dirs) != 20:
        raise ValueError(f"Expected 20 category directories, found {len(category_dirs)}")

    for category_dir in category_dirs:
        files = sorted(path for path in category_dir.iterdir() if path.is_file())
        for document_path in files:
            documents.append(document_path.read_text(encoding="latin-1", errors="replace"))
            labels.append(category_dir.name)

    if not documents:
        raise ValueError(f"No documents found in {dataset_dir}")

    label_ids = {name: index for index, name in enumerate(category_dir.name for category_dir in category_dirs)}
    return documents, np.array([label_ids[label] for label in labels], dtype=np.int32)


def make_pairs(
    features: np.ndarray,
    labels: np.ndarray,
    pairs_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Create balanced same-category and different-category document pairs."""
    by_category = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
    category_ids = np.array(sorted(by_category))
    pair_features: list[np.ndarray] = []
    pair_labels: list[int] = []

    for _ in range(pairs_per_class):
        category = int(rng.choice(category_ids))
        first, second = rng.choice(by_category[category], size=2, replace=False)
        pair_features.append(np.concatenate((features[first], features[second])))
        pair_labels.append(1)

        first_category, second_category = rng.choice(category_ids, size=2, replace=False)
        first = int(rng.choice(by_category[int(first_category)]))
        second = int(rng.choice(by_category[int(second_category)]))
        pair_features.append(np.concatenate((features[first], features[second])))
        pair_labels.append(0)

    order = rng.permutation(len(pair_labels))
    return np.asarray(pair_features, dtype=np.uint32)[order], np.asarray(pair_labels, dtype=np.int32)[order]


def evaluate(model: MultiClassTsetlinMachine, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """Calculate the requested binary classification metrics."""
    predictions = model.predict(features).astype(np.int32)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def inspect_clauses(model: MultiClassTsetlinMachine) -> dict[str, Any]:
    """Report the learned clause-state structure exposed by the library."""
    if not callable(getattr(model, "get_state", None)):
        return {"available": False, "reason": "The installed library exposes no get_state method."}

    class_states = model.get_state()

    return {
        "available": True,
        "classes": len(class_states),
        "state_shapes": [[list(np.asarray(array).shape) for array in class_state] for class_state in class_states],
        "negated_literals_enabled": True,
        "note": "The model was created with append_negated=True. The second state array per class contains "
        "the automata states used to learn included positive and negated literals.",
    }


def save_results(
    results_dir: Path,
    history: list[dict[str, float | int]],
    final_metrics: dict[str, float],
    clause_report: dict[str, Any],
) -> None:
    """Write metrics, clause information, and an accuracy plot."""
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (results_dir / "final_metrics.json").write_text(
        json.dumps({"metrics": final_metrics, "clause_inspection": clause_report}, indent=2),
        encoding="utf-8",
    )

    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["train_accuracy"] for row in history], label="Training accuracy")
    plt.plot(epochs, [row["test_accuracy"] for row in history], label="Testing accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "accuracy_by_epoch.png", dpi=150)
    plt.close()


def run_experiment(
    dataset_dir: Path,
    results_dir: Path,
    pairs_per_class: int,
    epochs: int,
    max_features: int,
) -> dict[str, float]:
    """Run the complete leakage-safe pair classification experiment."""
    random.seed(SEED)
    np.random.seed(SEED)
    documents, labels = load_local_newsgroups(dataset_dir)
    document_indices = np.arange(len(documents))
    train_indices, test_indices = train_test_split(
        document_indices, test_size=0.2, random_state=SEED, stratify=labels
    )

    vectorizer = CountVectorizer(binary=True, max_features=max_features, min_df=2, stop_words="english")
    train_words = vectorizer.fit_transform([documents[index] for index in train_indices]).toarray().astype(np.uint32)
    test_words = vectorizer.transform([documents[index] for index in test_indices]).toarray().astype(np.uint32)
    train_pairs, train_targets = make_pairs(train_words, labels[train_indices], pairs_per_class, np.random.default_rng(SEED))
    test_pairs, test_targets = make_pairs(test_words, labels[test_indices], pairs_per_class, np.random.default_rng(SEED + 1))

    model = MultiClassTsetlinMachine(number_of_clauses=100, T=15, s=1.0, append_negated=True)
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        # Continue training the same model rather than resetting it each epoch.
        model.fit(train_pairs, train_targets, epochs=1, incremental=True)
        train_metrics = evaluate(model, train_pairs, train_targets)
        test_metrics = evaluate(model, test_pairs, test_targets)
        history.append(
            {"epoch": epoch, "train_accuracy": train_metrics["accuracy"], "test_accuracy": test_metrics["accuracy"]}
        )
        print(f"Epoch {epoch}/{epochs}: train accuracy={train_metrics['accuracy']:.4f}, "
              f"test accuracy={test_metrics['accuracy']:.4f}")

    final_metrics = evaluate(model, test_pairs, test_targets)
    clause_report = inspect_clauses(model)
    save_results(results_dir, history, final_metrics, clause_report)
    print(json.dumps(final_metrics, indent=2))
    print(f"Results saved in {results_dir}")
    return final_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--pairs-per-class", type=int, default=2000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=1000)
    args = parser.parse_args()
    if min(args.pairs_per_class, args.epochs, args.max_features) <= 0:
        parser.error("pairs-per-class, epochs, and max-features must be positive")
    run_experiment(args.dataset, args.results, args.pairs_per_class, args.epochs, args.max_features)


if __name__ == "__main__":
    main()