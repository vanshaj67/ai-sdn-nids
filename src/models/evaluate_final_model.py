from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


TEST_FILE = Path("data/splits/test.csv")

MODEL_FILE = Path(
    "outputs/models/improved/improved_pipeline.joblib"
)

VALIDATION_METRICS_FILE = Path(
    "outputs/models/improved/validation_metrics.json"
)

OUTPUT_DIR = Path("outputs/models/final_test")

CHUNK_SIZE = 50_000


def load_test_data(feature_columns):
    """
    Load the test split in chunks and enforce the feature order used
    during training.
    """
    feature_chunks = []
    label_chunks = []
    total_rows = 0

    required_columns = feature_columns + ["Label"]

    print("Loading the untouched test dataset in chunks...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            TEST_FILE,
            chunksize=CHUNK_SIZE,
            dtype={"Label": "string"},
        ),
        start=1,
    ):
        missing_columns = [
            column
            for column in required_columns
            if column not in chunk.columns
        ]

        if missing_columns:
            raise ValueError(
                "The test dataset is missing required columns: "
                + ", ".join(missing_columns)
            )

        feature_chunk = chunk[feature_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )

        label_chunk = chunk["Label"].astype(str).to_numpy()

        feature_chunks.append(feature_chunk)
        label_chunks.append(label_chunk)

        total_rows += len(chunk)

        print(
            f"Loaded chunk {chunk_number}: "
            f"{len(chunk):,} rows "
            f"(total {total_rows:,})"
        )

    if not feature_chunks:
        raise ValueError("The test dataset contains no records.")

    x_test = np.concatenate(
        feature_chunks,
        axis=0,
    )

    test_labels = np.concatenate(
        label_chunks,
        axis=0,
    )

    print(f"Test rows loaded: {len(test_labels):,}")

    return x_test, test_labels


def calculate_metrics(
    actual_labels,
    predicted_labels,
    test_rows,
    feature_count,
    class_count,
    elapsed_seconds,
):
    """Calculate final test metrics."""
    return {
        "model": "ExtraTreesClassifier",
        "evaluation_split": "test",
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
        "test_rows": int(test_rows),
        "features": int(feature_count),
        "classes": int(class_count),
        "evaluation_seconds": float(elapsed_seconds),
        "model_retrained": False,
    }


def save_validation_comparison(test_metrics):
    """
    Compare the selected model's validation and final test results.
    This comparison does not change or retrain the model.
    """
    if not VALIDATION_METRICS_FILE.exists():
        print(
            "\nValidation metrics were not found; "
            "comparison file was not created."
        )
        return None

    with open(
        VALIDATION_METRICS_FILE,
        "r",
        encoding="utf-8",
    ) as validation_file:
        validation_metrics = json.load(validation_file)

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
        validation_value = float(
            validation_metrics[metric_name]
        )

        test_value = float(
            test_metrics[metric_name]
        )

        comparison_rows.append(
            {
                "metric": metric_name,
                "validation": validation_value,
                "test": test_value,
                "test_minus_validation": (
                    test_value - validation_value
                ),
            }
        )

    comparison = pd.DataFrame(comparison_rows)

    comparison.to_csv(
        OUTPUT_DIR / "validation_vs_test.csv",
        index=False,
    )

    return comparison


def main():
    start_time = time.time()

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_FILE}"
        )

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Improved model not found: {MODEL_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading the already-selected improved pipeline...")

    pipeline = joblib.load(MODEL_FILE)

    required_pipeline_items = {
        "model",
        "label_encoder",
        "feature_columns",
    }

    missing_pipeline_items = (
        required_pipeline_items - set(pipeline)
    )

    if missing_pipeline_items:
        raise ValueError(
            "The saved pipeline is missing: "
            + ", ".join(sorted(missing_pipeline_items))
        )

    model = pipeline["model"]
    label_encoder = pipeline["label_encoder"]
    feature_columns = pipeline["feature_columns"]

    print(f"Model: {model.__class__.__name__}")
    print(f"Expected features: {len(feature_columns)}")
    print(f"Known classes: {len(label_encoder.classes_)}")
    print("The model will not be retrained.")

    x_test, test_text_labels = load_test_data(
        feature_columns
    )

    known_classes = set(
        label_encoder.classes_.astype(str)
    )

    test_classes = set(test_text_labels)

    unknown_classes = sorted(
        test_classes - known_classes
    )

    if unknown_classes:
        raise ValueError(
            "The test set contains unknown classes: "
            + ", ".join(unknown_classes)
        )

    y_test = label_encoder.transform(
        test_text_labels
    )

    print("\nRunning the one-time final test prediction...")

    predictions = model.predict(x_test)

    elapsed_seconds = time.time() - start_time

    number_of_classes = len(
        label_encoder.classes_
    )

    classes = np.arange(
        number_of_classes
    )

    metrics = calculate_metrics(
        actual_labels=y_test,
        predicted_labels=predictions,
        test_rows=len(y_test),
        feature_count=len(feature_columns),
        class_count=number_of_classes,
        elapsed_seconds=elapsed_seconds,
    )

    report = classification_report(
        y_test,
        predictions,
        labels=classes,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR
        / "test_classification_report.csv"
    )

    matrix = confusion_matrix(
        y_test,
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
        OUTPUT_DIR / "test_confusion_matrix.csv"
    )

    test_class_distribution = pd.DataFrame(
        {
            "class_name": label_encoder.classes_,
            "test_count": np.bincount(
                y_test,
                minlength=number_of_classes,
            ),
        }
    )

    test_class_distribution.to_csv(
        OUTPUT_DIR / "test_class_distribution.csv",
        index=False,
    )

    with open(
        OUTPUT_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as metrics_file:
        json.dump(
            metrics,
            metrics_file,
            indent=4,
        )

    comparison = save_validation_comparison(
        metrics
    )

    print("\n" + "=" * 60)
    print("Final test evaluation completed successfully.")
    print(f"Test rows:         {metrics['test_rows']:,}")
    print(f"Accuracy:          {metrics['accuracy']:.6f}")
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
    print(f"Macro F1:          {metrics['macro_f1']:.6f}")
    print(
        f"Weighted F1:       "
        f"{metrics['weighted_f1']:.6f}"
    )
    print(
        f"Evaluation time:   "
        f"{metrics['evaluation_seconds']:.2f} seconds"
    )

    if comparison is not None:
        print("\nValidation-to-test changes:")

        for metric_name in [
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "weighted_f1",
        ]:
            row = comparison[
                comparison["metric"] == metric_name
            ].iloc[0]

            print(
                f"{metric_name:18s} "
                f"{row['test_minus_validation']:+.6f}"
            )

    print(f"\nOutputs: {OUTPUT_DIR}")
    print("The saved model was evaluated without retraining.")


if __name__ == "__main__":
    main()
