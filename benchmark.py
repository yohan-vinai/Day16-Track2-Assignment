#!/usr/bin/env python3
"""Train and measure a LightGBM credit-card fraud classifier for Lab Day 16."""

import argparse
import json
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path.home() / "ml-benchmark" / "creditcard.csv",
        help="Path to Kaggle's creditcard.csv (default: ~/ml-benchmark/creditcard.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_result.json"),
        help="Where to write the JSON metrics (default: ./benchmark_result.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "Download it first with: kaggle datasets download -d "
            "mlg-ulb/creditcardfraud --unzip -p ~/ml-benchmark/"
        )

    load_started = time.perf_counter()
    data = pd.read_csv(args.data)
    if "Class" not in data.columns:
        raise SystemExit("Expected a 'Class' target column in the dataset.")
    X = data.drop(columns="Class")
    y = data["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    load_seconds = time.perf_counter() - load_started

    fraud_count = int((y_train == 1).sum())
    legitimate_count = int((y_train == 0).sum())
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=legitimate_count / fraud_count,
        random_state=42,
        n_jobs=2,
        verbosity=-1,
    )

    training_started = time.perf_counter()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )
    training_seconds = time.perf_counter() - training_started

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype("int8")

    # Repeat single-row prediction to reduce timer noise on a fast CPU model.
    one_row = X_test.iloc[[0]]
    single_runs = 100
    single_started = time.perf_counter()
    for _ in range(single_runs):
        model.predict_proba(one_row)
    single_latency_ms = (time.perf_counter() - single_started) * 1000 / single_runs

    batch = X_test.iloc[: min(1000, len(X_test))]
    batch_started = time.perf_counter()
    model.predict_proba(batch)
    batch_seconds = time.perf_counter() - batch_started
    throughput_rows_per_second = len(batch) / batch_seconds if batch_seconds else None

    results = {
        "dataset": str(args.data),
        "rows": int(len(data)),
        "features": int(X.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "load_time_seconds": load_seconds,
        "training_time_seconds": training_seconds,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "auc_roc": roc_auc_score(y_test, probabilities),
        "accuracy": accuracy_score(y_test, predictions),
        "f1_score": f1_score(y_test, predictions, zero_division=0),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "inference_latency_one_row_ms_mean_of_100": single_latency_ms,
        "inference_throughput_rows_per_second_batch_of_1000": throughput_rows_per_second,
        "inference_batch_rows": int(len(batch)),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"\nSaved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
