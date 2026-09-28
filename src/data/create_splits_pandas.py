from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = Path("data/processed/cicids2017_clean.csv")
SPLIT_DIR = Path("data/splits")
REPORT_DIR = Path("outputs/data_preparation")

TRAIN_FILE = SPLIT_DIR / "train.csv"
VALIDATION_FILE = SPLIT_DIR / "validation.csv"
TEST_FILE = SPLIT_DIR / "test.csv"

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


def save_class_distribution(dataframe, filename):
    """Save class counts and percentages for one dataset split."""
    counts = dataframe["Label"].value_counts().sort_index()

    distribution = pd.DataFrame(
        {
            "label": counts.index,
            "count": counts.values,
            "percentage": counts.values / len(dataframe) * 100,
        }
    )

    distribution.to_csv(REPORT_DIR / filename, index=False)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Clean dataset not found: {INPUT_FILE}")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {INPUT_FILE}")
    data = pd.read_csv(INPUT_FILE, low_memory=False)

    if "Label" not in data.columns:
        raise KeyError("Label column is missing from the clean dataset.")

    original_rows = len(data)
    feature_columns = [
        column for column in data.columns if column != "Label"
    ]

    print(f"Loaded {original_rows:,} rows.")
    print(f"Initial features: {len(feature_columns)}")

    constant_columns = [
        column
        for column in feature_columns
        if data[column].nunique(dropna=False) <= 1
    ]

    data.drop(columns=constant_columns, inplace=True)

    feature_columns = [
        column for column in data.columns if column != "Label"
    ]

    print(f"Constant features removed: {len(constant_columns)}")
    print(f"Remaining features: {len(feature_columns)}")

    print("Finding identical feature vectors and label conflicts...")

    data["_feature_hash"] = pd.util.hash_pandas_object(
        data[feature_columns],
        index=False,
    )

    labels_per_hash = data.groupby("_feature_hash")["Label"].nunique()
    conflicting_hashes = labels_per_hash[
        labels_per_hash > 1
    ].index

    conflicting_mask = data["_feature_hash"].isin(conflicting_hashes)
    conflicting_rows = int(conflicting_mask.sum())
    conflicting_groups = len(conflicting_hashes)

    data = data.loc[~conflicting_mask].copy()

    rows_before_deduplication = len(data)

    data.drop_duplicates(
        subset=feature_columns,
        keep="first",
        inplace=True,
    )

    duplicate_rows_removed = rows_before_deduplication - len(data)

    data.drop(columns="_feature_hash", inplace=True)
    data.reset_index(drop=True, inplace=True)

    prepared_rows = len(data)

    print(f"Conflicting groups removed: {conflicting_groups:,}")
    print(f"Conflicting rows removed: {conflicting_rows:,}")
    print(f"Duplicate rows removed: {duplicate_rows_removed:,}")
    print(f"Reliable unique rows: {prepared_rows:,}")

    print("\nCreating stratified 70/15/15 splits...")

    train_data, temporary_data = train_test_split(
        data,
        test_size=VALIDATION_SIZE + TEST_SIZE,
        stratify=data["Label"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=0.50,
        stratify=temporary_data["Label"],
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_data = train_data.reset_index(drop=True)
    validation_data = validation_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)

    print(f"Training rows:   {len(train_data):,}")
    print(f"Validation rows: {len(validation_data):,}")
    print(f"Test rows:       {len(test_data):,}")

    print("\nSaving dataset splits...")

    train_data.to_csv(TRAIN_FILE, index=False)
    validation_data.to_csv(VALIDATION_FILE, index=False)
    test_data.to_csv(TEST_FILE, index=False)

    save_class_distribution(
        train_data,
        "train_class_distribution.csv",
    )
    save_class_distribution(
        validation_data,
        "validation_class_distribution.csv",
    )
    save_class_distribution(
        test_data,
        "test_class_distribution.csv",
    )

    pd.DataFrame(
        {
            "removed_constant_feature": constant_columns
        }
    ).to_csv(
        REPORT_DIR / "removed_constant_features.csv",
        index=False,
    )

    summary = {
        "random_state": RANDOM_STATE,
        "original_rows": original_rows,
        "constant_features_removed": len(constant_columns),
        "remaining_features": len(feature_columns),
        "conflicting_feature_groups_removed": conflicting_groups,
        "conflicting_rows_removed": conflicting_rows,
        "duplicate_rows_removed": duplicate_rows_removed,
        "reliable_unique_rows": prepared_rows,
        "training_rows": len(train_data),
        "validation_rows": len(validation_data),
        "test_rows": len(test_data),
        "training_percentage": TRAIN_SIZE * 100,
        "validation_percentage": VALIDATION_SIZE * 100,
        "test_percentage": TEST_SIZE * 100,
    }

    with open(
        REPORT_DIR / "preparation_summary.json",
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(summary, summary_file, indent=4)

    print("=" * 60)
    print("Data preparation completed successfully.")
    print(f"Training dataset:   {TRAIN_FILE}")
    print(f"Validation dataset: {VALIDATION_FILE}")
    print(f"Test dataset:       {TEST_FILE}")
    print(f"Reports directory:  {REPORT_DIR}")


if __name__ == "__main__":
    main()
