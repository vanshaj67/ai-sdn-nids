from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/external/cicids2017/MachineLearningCVE")
OUTPUT_DIR = Path("outputs/dataset_inspection")
CHUNK_SIZE = 100_000


def clean_label(label):
    """Standardise inconsistent CIC-IDS2017 label names."""
    label = str(label).strip()

    replacements = {
        "Web Attack � Brute Force": "Web Attack - Brute Force",
        "Web Attack � XSS": "Web Attack - XSS",
        "Web Attack � Sql Injection": "Web Attack - SQL Injection",
    }

    return replacements.get(label, label)


def inspect_dataset():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    file_summaries = []
    label_counts = {}
    missing_counts = {}
    infinite_counts = {}

    print(f"Found {len(csv_files)} CSV files.\n")

    for csv_file in csv_files:
        print(f"Inspecting: {csv_file.name}")

        file_rows = 0
        file_columns = 0

        for chunk in pd.read_csv(csv_file, chunksize=CHUNK_SIZE):
            chunk.columns = chunk.columns.str.strip()
            file_rows += len(chunk)
            file_columns = len(chunk.columns)

            if "Label" not in chunk.columns:
                raise KeyError(f"Label column not found in {csv_file.name}")

            chunk["Label"] = chunk["Label"].apply(clean_label)

            counts = chunk["Label"].value_counts()
            for label, count in counts.items():
                label_counts[label] = label_counts.get(label, 0) + int(count)

            missing = chunk.isna().sum()
            for column, count in missing.items():
                missing_counts[column] = (
                    missing_counts.get(column, 0) + int(count)
                )

            numeric_chunk = chunk.select_dtypes(include=[np.number])
            infinite = np.isinf(numeric_chunk).sum()

            for column, count in infinite.items():
                infinite_counts[column] = (
                    infinite_counts.get(column, 0) + int(count)
                )

        file_summaries.append(
            {
                "file": csv_file.name,
                "rows": file_rows,
                "columns": file_columns,
            }
        )

        print(f"  Rows: {file_rows:,}")
        print(f"  Columns: {file_columns}\n")

    pd.DataFrame(file_summaries).to_csv(
        OUTPUT_DIR / "file_summary.csv", index=False
    )

    pd.Series(label_counts, name="count").sort_values(
        ascending=False
    ).to_csv(OUTPUT_DIR / "label_counts.csv")

    pd.Series(missing_counts, name="missing_count").sort_values(
        ascending=False
    ).to_csv(OUTPUT_DIR / "missing_values.csv")

    pd.Series(infinite_counts, name="infinite_count").sort_values(
        ascending=False
    ).to_csv(OUTPUT_DIR / "infinite_values.csv")

    total_rows = sum(item["rows"] for item in file_summaries)

    print("=" * 60)
    print(f"Inspection completed successfully.")
    print(f"Total records: {total_rows:,}")
    print(f"Unique labels: {len(label_counts)}")
    print(f"Reports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    inspect_dataset()
