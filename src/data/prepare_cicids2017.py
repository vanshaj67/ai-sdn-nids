from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("data/external/cicids2017/MachineLearningCVE")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "cicids2017_clean.csv"
SUMMARY_FILE = OUTPUT_DIR / "cleaning_summary.csv"
CHUNK_SIZE = 100_000


def clean_label(label):
    """Standardise CIC-IDS2017 label names."""
    label = str(label).strip()

    replacements = {
        "Web Attack � Brute Force": "Web Attack - Brute Force",
        "Web Attack � XSS": "Web Attack - XSS",
        "Web Attack � Sql Injection": "Web Attack - SQL Injection",
        "Web Attack – Brute Force": "Web Attack - Brute Force",
        "Web Attack – XSS": "Web Attack - XSS",
        "Web Attack – Sql Injection": "Web Attack - SQL Injection",
    }

    return replacements.get(label, label)


def prepare_dataset():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {DATA_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    summary_rows = []
    total_input = 0
    total_removed = 0
    total_output = 0
    write_header = True

    print(f"Found {len(csv_files)} CSV files.")
    print("Starting CIC-IDS2017 cleaning...\n")

    for csv_file in csv_files:
        print(f"Processing: {csv_file.name}")

        file_input = 0
        file_removed = 0
        file_output = 0

        for chunk in pd.read_csv(
            csv_file,
            chunksize=CHUNK_SIZE,
            low_memory=False
        ):
            chunk.columns = chunk.columns.str.strip()

            if "Label" not in chunk.columns:
                raise KeyError(f"Label column not found in {csv_file.name}")

            input_rows = len(chunk)
            file_input += input_rows
            total_input += input_rows

            chunk["Label"] = chunk["Label"].apply(clean_label)

            feature_columns = [
                column for column in chunk.columns if column != "Label"
            ]

            chunk[feature_columns] = chunk[feature_columns].apply(
                pd.to_numeric,
                errors="coerce"
            )

            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)

            chunk.dropna(subset=feature_columns + ["Label"], inplace=True)

            removed_rows = input_rows - len(chunk)
            output_rows = len(chunk)

            file_removed += removed_rows
            total_removed += removed_rows
            file_output += output_rows
            total_output += output_rows

            chunk.to_csv(
                OUTPUT_FILE,
                mode="w" if write_header else "a",
                header=write_header,
                index=False
            )

            write_header = False

        summary_rows.append(
            {
                "file": csv_file.name,
                "input_rows": file_input,
                "removed_invalid_rows": file_removed,
                "output_rows": file_output,
            }
        )

        print(f"  Input rows:   {file_input:,}")
        print(f"  Removed rows: {file_removed:,}")
        print(f"  Output rows:  {file_output:,}\n")

    pd.DataFrame(summary_rows).to_csv(SUMMARY_FILE, index=False)

    removed_percentage = (
        total_removed / total_input * 100 if total_input else 0
    )

    print("=" * 60)
    print("Dataset preparation completed successfully.")
    print(f"Total input records:   {total_input:,}")
    print(f"Invalid records removed: {total_removed:,}")
    print(f"Records retained:      {total_output:,}")
    print(f"Percentage removed:    {removed_percentage:.4f}%")
    print(f"Clean dataset:         {OUTPUT_FILE}")
    print(f"Cleaning summary:      {SUMMARY_FILE}")


if __name__ == "__main__":
    prepare_dataset()
