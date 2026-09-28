from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder


TRAIN_FILE = Path("data/splits/train.csv")
VALIDATION_FILE = Path("data/splits/validation.csv")

BASELINE_METRICS_FILE = Path(
    "outputs/models/baseline/validation_metrics.json"
)

OUTPUT_DIR = Path("outputs/models/improved")

CHUNK_SIZE = 50_000
MAX_SAMPLES_PER_CLASS = 30_000
RANDOM_STATE = 42

NUMBER_OF_TREES = 150
MAX_DEPTH = 30
MIN_SAMPLES_SPLIT = 4
MIN_SAMPLES_LEAF = 2
MAX_FEATURES = "sqrt"
N_JOBS = 2


def deterministic_scores(dataframe):
    """
    Create deterministic sampling scores from complete rows.

    Keeping the rows with the smallest scores provides a reproducible
    per-class sample without loading the entire training CSV into RAM.
    """
    hashes = pd.util.hash_pandas_object(
        dataframe,
        index=False,
        hash_key="0123456789abcdef",
    ).to_numpy(dtype=np.uint64)

    return hashes


def load_balanced_training_sample():
    """
    Read training data in chunks and retain up to a fixed number of
    deterministic samples from every class.
    """
    samples_by_class = {}
    complete_class_counts = {}

    print("Loading a balanced training sample in chunks...")

    reader = pd.read_csv(
        TRAIN_FILE,
        chunksize=CHUNK_SIZE,
        dtype={"Label": "string"},
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        for class_name, class_data in chunk.groupby(
            "Label",
            sort=False,
            observed=True,
        ):
            class_name = str(class_name)
            class_data = class_data.copy()

            complete_class_counts[class_name] = (
                complete_class_counts.get(class_name, 0)
                + len(class_data)
            )

            class_data["_sampling_score"] = (
                deterministic_scores(class_data)
            )

            if class_name in samples_by_class:
                candidates = pd.concat(
                    [
                        samples_by_class[class_name],
                        class_data,
                    ],
                    ignore_index=True,
                )
            else:
                candidates = class_data

            if len(candidates) > MAX_SAMPLES_PER_CLASS:
                candidates = candidates.nsmallest(
                    MAX_SAMPLES_PER_CLASS,
                    "_sampling_score",
                    keep="first",
                )

            samples_by_class[class_name] = candidates

        if chunk_number % 5 == 0:
            rows_processed = chunk_number * CHUNK_SIZE
            retained_rows = sum(
                len(data)
                for data in samples_by_class.values()
            )

            print(
                f"Processed approximately {rows_processed:,} rows; "
                f"retained {retained_rows:,} rows."
            )

    sampled_frames = []

    for class_name in sorted(samples_by_class):
        class_sample = samples_by_class[class_name].drop(
            columns="_sampling_score"
        )
        sampled_frames.append(class_sample)

    training_data = pd.concat(
        sampled_frames,
        ignore_index=True,
    )

    training_data = training_data.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(
        f"Balanced training sample rows: "
        f"{len(training_data):,}"
    )
    print(
        f"Classes retained: "
        f"{training_data['Label'].nunique()}"
    )

    return training_data, complete_class_counts


def load_validation_data():
    """Load the complete validation split."""
    print("\nLoading the complete validation dataset...")

    validation_data = pd.read_csv(
        VALIDATION_FILE,
        dtype={"Label": "string"},
    )

    print(f"Validation rows: {len(validation_data):,}")

    return validation_data


def calculate_metrics(
    actual_labels,
    predicted_labels,
    training_rows,
    validation_rows,
    feature_count,
    class_count,
    elapsed_seconds,
):
    """Calculate overall validation metrics."""
    return {
        "model": "ExtraTreesClassifier",
        "accuracy": float(
            accuracy_score(
                actual_labels,
                predicted_labels,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                actual_labels,
                predicted_labels,
            )
        ),
        "macro_precision": float(
            precision_score(
                actual_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                actual_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                actual_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                actual_labels,
                predicted_labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "training_sample_rows": int(training_rows),
        "validation_rows": int(validation_rows),
        "features": int(feature_count),
        "classes": int(class_count),
        "maximum_samples_per_class": (
            MAX_SAMPLES_PER_CLASS
        ),
        "number_of_trees": NUMBER_OF_TREES,
        "max_depth": MAX_DEPTH,
        "min_samples_split": MIN_SAMPLES_SPLIT,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "max_features": MAX_FEATURES,
        "random_state": RANDOM_STATE,
        "training_and_evaluation_seconds": float(
            elapsed_seconds
        ),
    }


def save_comparison(improved_metrics):
    """Compare improved validation results with the baseline."""
    if not BASELINE_METRICS_FILE.exists():
        print(
            "\nBaseline metrics were not found; "
            "comparison file was not created."
        )
        return None

    with open(
        BASELINE_METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as baseline_file:
        baseline_metrics = json.load(baseline_file)

    metric_names = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
    ]

    comparison_rows = []

    for metric_name in metric_names:
        baseline_value = float(
            baseline_metrics[metric_name]
        )
        improved_value = float(
            improved_metrics[metric_name]
        )

        comparison_rows.append(
            {
                "metric": metric_name,
                "baseline": baseline_value,
                "improved": improved_value,
                "absolute_change": (
                    improved_value - baseline_value
                ),
                "improved_model_better": (
                    improved_value > baseline_value
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows)

    comparison.to_csv(
        OUTPUT_DIR / "baseline_vs_improved.csv",
        index=False,
    )

    return comparison


def main():
    start_time = time.time()

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: "
            f"{VALIDATION_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_data, complete_class_counts = (
        load_balanced_training_sample()
    )

    validation_data = load_validation_data()

    feature_columns = [
        column
        for column in training_data.columns
        if column != "Label"
    ]

    training_classes = set(
        training_data["Label"].astype(str).unique()
    )
    validation_classes = set(
        validation_data["Label"].astype(str).unique()
    )

    missing_classes = sorted(
        validation_classes - training_classes
    )

    if missing_classes:
        raise ValueError(
            "Training data is missing validation classes: "
            + ", ".join(missing_classes)
        )

    print(f"Features: {len(feature_columns)}")
    print(f"Classes: {len(training_classes)}")

    print("\nBalanced training sample class counts:")
    print(
        training_data["Label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    label_encoder = LabelEncoder()

    label_encoder.fit(
        training_data["Label"].astype(str)
    )

    y_train = label_encoder.transform(
        training_data["Label"].astype(str)
    )

    y_validation = label_encoder.transform(
        validation_data["Label"].astype(str)
    )

    print("\nConverting features to float32 arrays...")

    x_train = training_data[feature_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    x_validation = validation_data[
        feature_columns
    ].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    sampled_class_counts = (
        training_data["Label"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    del training_data
    del validation_data

    number_of_classes = len(
        label_encoder.classes_
    )

    print("\nTraining the nonlinear Extra Trees model...")
    print(f"Trees: {NUMBER_OF_TREES}")
    print(f"Maximum depth: {MAX_DEPTH}")
    print(f"CPU threads: {N_JOBS}")

    model = ExtraTreesClassifier(
        n_estimators=NUMBER_OF_TREES,
        criterion="gini",
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features=MAX_FEATURES,
        class_weight="balanced",
        bootstrap=False,
        n_jobs=N_JOBS,
        random_state=RANDOM_STATE,
        verbose=1,
    )

    model.fit(
        x_train,
        y_train,
    )

    print("\nPredicting the complete validation set...")

    predictions = model.predict(
        x_validation
    )

    elapsed_seconds = time.time() - start_time

    metrics = calculate_metrics(
        actual_labels=y_validation,
        predicted_labels=predictions,
        training_rows=len(y_train),
        validation_rows=len(y_validation),
        feature_count=len(feature_columns),
        class_count=number_of_classes,
        elapsed_seconds=elapsed_seconds,
    )

    classes = np.arange(
        number_of_classes
    )

    report = classification_report(
        y_validation,
        predictions,
        labels=classes,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR
        / "validation_classification_report.csv"
    )

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=classes,
    )

    confusion_dataframe = pd.DataFrame(
        matrix,
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    )

    confusion_dataframe.index.name = "actual_label"

    confusion_dataframe.to_csv(
        OUTPUT_DIR
        / "validation_confusion_matrix.csv"
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    feature_importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
    )

    sampling_rows = []

    for class_name in label_encoder.classes_:
        sampling_rows.append(
            {
                "class_name": class_name,
                "complete_training_count": (
                    complete_class_counts.get(
                        class_name,
                        0,
                    )
                ),
                "sampled_training_count": (
                    sampled_class_counts.get(
                        class_name,
                        0,
                    )
                ),
            }
        )

    pd.DataFrame(sampling_rows).to_csv(
        OUTPUT_DIR / "training_sampling_report.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "validation_metrics.json",
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=4,
        )

    comparison = save_comparison(
        metrics
    )

    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
        },
        OUTPUT_DIR / "improved_pipeline.joblib",
        compress=3,
    )

    print("\n" + "=" * 60)
    print("Improved model training completed successfully.")
    print(
        f"Accuracy:          "
        f"{metrics['accuracy']:.6f}"
    )
    print(
        f"Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.6f}"
    )
    print(
        f"Macro precision:   "
        f"{metrics['macro_precision']:.6f}"
    )
    print(
        f"Macro recall:      "
        f"{metrics['macro_recall']:.6f}"
    )
    print(
        f"Macro F1:          "
        f"{metrics['macro_f1']:.6f}"
    )
    print(
        f"Weighted F1:       "
        f"{metrics['weighted_f1']:.6f}"
    )
    print(
        f"Runtime:           "
        f"{metrics['training_and_evaluation_seconds']:.2f} "
        f"seconds"
    )

    if comparison is not None:
        macro_row = comparison[
            comparison["metric"] == "macro_f1"
        ].iloc[0]

        balanced_row = comparison[
            comparison["metric"]
            == "balanced_accuracy"
        ].iloc[0]

        print("\nComparison with baseline:")
        print(
            f"Macro F1 change:          "
            f"{macro_row['absolute_change']:+.6f}"
        )
        print(
            f"Balanced accuracy change: "
            f"{balanced_row['absolute_change']:+.6f}"
        )

    print(f"\nOutputs: {OUTPUT_DIR}")
    print(
        "The test dataset has not been used."
    )


if __name__ == "__main__":
    main()
