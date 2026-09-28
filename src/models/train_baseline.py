from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler


TRAIN_FILE = Path("data/splits/train.csv")
VALIDATION_FILE = Path("data/splits/validation.csv")
OUTPUT_DIR = Path("outputs/models/baseline")

CHUNK_SIZE = 50_000
SAMPLE_FRACTION = 0.25
RARE_CLASS_THRESHOLD = 1_000
RANDOM_STATE = 42
EPOCHS = 5


def load_training_sample():
    """
    Load a deterministic training sample in chunks.

    All records from rare classes are retained, while larger classes
    are sampled to keep memory usage manageable.
    """
    sampled_chunks = []

    print("Loading a memory-efficient training sample...")

    for chunk_number, chunk in enumerate(
        pd.read_csv(TRAIN_FILE, chunksize=CHUNK_SIZE)
    ):
        chunk_samples = []

        for _, class_data in chunk.groupby("Label", sort=False):
            if len(class_data) < RARE_CLASS_THRESHOLD:
                # Preserve rare classes completely.
                sampled_class = class_data
            else:
                sampled_class = class_data.sample(
                    frac=SAMPLE_FRACTION,
                    random_state=RANDOM_STATE + chunk_number,
                )

            chunk_samples.append(sampled_class)

        sampled_chunks.append(
            pd.concat(chunk_samples, ignore_index=True)
        )

    training_data = pd.concat(sampled_chunks, ignore_index=True)

    print(f"Sampled training rows: {len(training_data):,}")
    return training_data


def load_validation_data():
    """Load the complete validation split."""
    print("Loading the complete validation dataset...")

    validation_data = pd.read_csv(VALIDATION_FILE)

    print(f"Validation rows: {len(validation_data):,}")
    return validation_data


def save_confusion_matrix(matrix, labels):
    """Save the validation confusion matrix as a CSV file."""
    confusion_dataframe = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )

    confusion_dataframe.index.name = "actual_label"

    confusion_dataframe.to_csv(
        OUTPUT_DIR / "validation_confusion_matrix.csv"
    )


def main():
    start_time = time.time()

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {VALIDATION_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    training_data = load_training_sample()
    validation_data = load_validation_data()

    feature_columns = [
        column
        for column in training_data.columns
        if column != "Label"
    ]

    training_classes = set(training_data["Label"].unique())
    validation_classes = set(validation_data["Label"].unique())

    missing_classes = sorted(
        validation_classes - training_classes
    )

    if missing_classes:
        raise ValueError(
            "The training sample is missing validation classes: "
            + ", ".join(missing_classes)
        )

    print(f"Features: {len(feature_columns)}")
    print(f"Classes: {training_data['Label'].nunique()}")

    print("\nTraining sample class counts:")
    print(
        training_data["Label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    label_encoder = LabelEncoder()
    label_encoder.fit(training_data["Label"])

    y_train = label_encoder.transform(
        training_data["Label"]
    )

    y_validation = label_encoder.transform(
        validation_data["Label"]
    )

    x_train = training_data[feature_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    x_validation = validation_data[feature_columns].to_numpy(
        dtype=np.float32,
        copy=True,
    )

    del training_data
    del validation_data

    print("\nFitting StandardScaler on training data only...")

    scaler = StandardScaler()

    x_train = scaler.fit_transform(x_train).astype(
        np.float32,
        copy=False,
    )

    x_validation = scaler.transform(x_validation).astype(
        np.float32,
        copy=False,
    )

    number_of_classes = len(label_encoder.classes_)
    number_of_samples = len(y_train)

    class_counts = np.bincount(
        y_train,
        minlength=number_of_classes,
    )

    if np.any(class_counts == 0):
        missing_indices = np.where(class_counts == 0)[0]
        missing_names = label_encoder.inverse_transform(
            missing_indices
        )

        raise ValueError(
            "No training samples were found for: "
            + ", ".join(missing_names)
        )

    class_weights = {
        class_index: (
            number_of_samples
            / (number_of_classes * class_count)
        )
        for class_index, class_count in enumerate(class_counts)
    }

    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.0001,
        class_weight=class_weights,
        random_state=RANDOM_STATE,
        max_iter=1,
        tol=None,
        average=True,
    )

    classes = np.arange(number_of_classes)
    random_generator = np.random.default_rng(
        RANDOM_STATE
    )

    print(
        f"\nTraining SGD baseline for {EPOCHS} epochs..."
    )

    for epoch in range(EPOCHS):
        shuffled_indices = random_generator.permutation(
            len(x_train)
        )

        model.partial_fit(
            x_train[shuffled_indices],
            y_train[shuffled_indices],
            classes=classes,
        )

        epoch_predictions = model.predict(x_validation)

        epoch_macro_f1 = f1_score(
            y_validation,
            epoch_predictions,
            average="macro",
            zero_division=0,
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} - "
            f"validation macro F1: {epoch_macro_f1:.6f}"
        )

    print("\nCalculating final validation metrics...")

    predictions = model.predict(x_validation)

    metrics = {
        "accuracy": float(
            accuracy_score(y_validation, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_validation,
                predictions,
            )
        ),
        "macro_precision": float(
            precision_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_validation,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_validation,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "training_sample_rows": int(len(y_train)),
        "validation_rows": int(len(y_validation)),
        "features": int(len(feature_columns)),
        "classes": int(number_of_classes),
        "sample_fraction_for_large_classes": (
            SAMPLE_FRACTION
        ),
        "rare_class_threshold_per_chunk": (
            RARE_CLASS_THRESHOLD
        ),
        "epochs": EPOCHS,
        "random_state": RANDOM_STATE,
        "training_seconds": float(
            time.time() - start_time
        ),
    }

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

    save_confusion_matrix(
        matrix,
        label_encoder.classes_,
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

    pd.DataFrame(
        {
            "encoded_label": classes,
            "class_name": label_encoder.classes_,
            "training_sample_count": class_counts,
            "class_weight": [
                class_weights[class_index]
                for class_index in classes
            ],
        }
    ).to_csv(
        OUTPUT_DIR / "class_mapping_and_weights.csv",
        index=False,
    )

    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
        },
        OUTPUT_DIR / "baseline_pipeline.joblib",
    )

    print("=" * 60)
    print("Baseline training completed successfully.")
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
    print(f"Outputs:           {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
