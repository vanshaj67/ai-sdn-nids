from pathlib import Path
import json

import duckdb


INPUT_FILE = Path("data/processed/cicids2017_clean.csv")
SPLIT_DIR = Path("data/splits")
REPORT_DIR = Path("outputs/data_preparation")
DATABASE_FILE = Path("data/processed/split_preparation.duckdb")
TEMP_DIR = Path("data/processed/duckdb_temp")

TRAIN_FILE = SPLIT_DIR / "train.csv"
VALIDATION_FILE = SPLIT_DIR / "validation.csv"
TEST_FILE = SPLIT_DIR / "test.csv"

MEMORY_LIMIT = "2GB"
THREADS = 2


def quote_identifier(name):
    """Safely quote a DuckDB column name."""
    return '"' + name.replace('"', '""') + '"'


def sql_path(path):
    """Return a path safe for use inside SQL strings."""
    return str(path.resolve()).replace("'", "''")


def save_distribution(connection, table_name, output_name):
    output_path = REPORT_DIR / output_name

    connection.execute(
        f"""
        COPY (
            SELECT
                Label AS label,
                COUNT(*) AS count,
                ROUND(
                    COUNT(*) * 100.0 /
                    SUM(COUNT(*)) OVER (),
                    6
                ) AS percentage
            FROM {table_name}
            GROUP BY Label
            ORDER BY Label
        )
        TO '{sql_path(output_path)}'
        (HEADER, DELIMITER ',');
        """
    )


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Clean dataset not found: {INPUT_FILE}")

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_DIR.mkdir(parents=True, exist_ok=True)

    print("Starting low-memory DuckDB data preparation...")
    print(f"Input file: {INPUT_FILE}")
    print(f"Memory limit: {MEMORY_LIMIT}")
    print(f"Threads: {THREADS}\n")

    connection = duckdb.connect(str(DATABASE_FILE))

    connection.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    connection.execute(f"SET threads = {THREADS}")
    connection.execute(
        f"SET temp_directory = '{sql_path(TEMP_DIR)}'"
    )
    connection.execute("SET preserve_insertion_order = false")

    input_path = sql_path(INPUT_FILE)

    print("Step 1/6: Reading the clean CSV structure...")

    description = connection.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_csv_auto(
            '{input_path}',
            header=true,
            sample_size=100000,
            all_varchar=false
        );
        """
    ).fetchall()

    all_columns = [row[0] for row in description]

    if "Label" not in all_columns:
        raise KeyError("Label column is missing from the clean dataset.")

    feature_columns = [
        column for column in all_columns if column != "Label"
    ]

    print(f"Detected features: {len(feature_columns)}")

    print("\nStep 2/6: Finding constant features...")

    constant_checks = []

    for column in feature_columns:
        quoted = quote_identifier(column)
        constant_checks.append(
            f"COUNT(DISTINCT {quoted}) <= 1 AS "
            f"{quote_identifier(column)}"
        )

    constant_result = connection.execute(
        f"""
        SELECT {", ".join(constant_checks)}
        FROM read_csv_auto(
            '{input_path}',
            header=true,
            sample_size=100000,
            all_varchar=false
        );
        """
    ).fetchone()

    constant_columns = [
        column
        for column, is_constant
        in zip(feature_columns, constant_result)
        if is_constant
    ]

    remaining_features = [
        column
        for column in feature_columns
        if column not in constant_columns
    ]

    print(f"Constant features removed: {len(constant_columns)}")
    print(f"Remaining features: {len(remaining_features)}")

    quoted_features = [
        quote_identifier(column)
        for column in remaining_features
    ]

    feature_list = ", ".join(quoted_features)

    print("\nStep 3/6: Grouping identical feature vectors...")
    print("This is the longest disk-backed step.")

    connection.execute("DROP TABLE IF EXISTS feature_groups")

    connection.execute(
        f"""
        CREATE TABLE feature_groups AS
        SELECT
            {feature_list},
            ANY_VALUE(Label) AS Label,
            COUNT(*) AS group_rows,
            COUNT(DISTINCT Label) AS label_count
        FROM read_csv_auto(
            '{input_path}',
            header=true,
            sample_size=100000,
            all_varchar=false
        )
        GROUP BY {feature_list};
        """
    )

    original_rows = connection.execute(
        "SELECT SUM(group_rows) FROM feature_groups"
    ).fetchone()[0]

    conflicting_groups, conflicting_rows = connection.execute(
        """
        SELECT
            COUNT(*),
            COALESCE(SUM(group_rows), 0)
        FROM feature_groups
        WHERE label_count > 1
        """
    ).fetchone()

    valid_rows_before_deduplication = connection.execute(
        """
        SELECT COALESCE(SUM(group_rows), 0)
        FROM feature_groups
        WHERE label_count = 1
        """
    ).fetchone()[0]

    prepared_rows = connection.execute(
        """
        SELECT COUNT(*)
        FROM feature_groups
        WHERE label_count = 1
        """
    ).fetchone()[0]

    duplicate_rows_removed = (
        valid_rows_before_deduplication - prepared_rows
    )

    print(f"Original rows: {original_rows:,}")
    print(f"Conflicting groups removed: {conflicting_groups:,}")
    print(f"Conflicting rows removed: {conflicting_rows:,}")
    print(f"Duplicate rows removed: {duplicate_rows_removed:,}")
    print(f"Reliable unique rows: {prepared_rows:,}")

    print("\nStep 4/6: Creating stratified 70/15/15 splits...")

    hash_arguments = ", ".join(quoted_features)

    connection.execute("DROP TABLE IF EXISTS prepared_ranked")

    connection.execute(
        f"""
        CREATE TABLE prepared_ranked AS
        SELECT
            {feature_list},
            Label,
            ROW_NUMBER() OVER (
                PARTITION BY Label
                ORDER BY hash({hash_arguments})
            ) AS class_row_number,
            COUNT(*) OVER (
                PARTITION BY Label
            ) AS class_total
        FROM feature_groups
        WHERE label_count = 1;
        """
    )

    connection.execute("DROP VIEW IF EXISTS training_data")
    connection.execute("DROP VIEW IF EXISTS validation_data")
    connection.execute("DROP VIEW IF EXISTS test_data")

    output_columns = feature_list + ", Label"

    connection.execute(
        f"""
        CREATE VIEW training_data AS
        SELECT {output_columns}
        FROM prepared_ranked
        WHERE class_row_number <= FLOOR(class_total * 0.70);
        """
    )

    connection.execute(
        f"""
        CREATE VIEW validation_data AS
        SELECT {output_columns}
        FROM prepared_ranked
        WHERE class_row_number > FLOOR(class_total * 0.70)
          AND class_row_number <= FLOOR(class_total * 0.85);
        """
    )

    connection.execute(
        f"""
        CREATE VIEW test_data AS
        SELECT {output_columns}
        FROM prepared_ranked
        WHERE class_row_number > FLOOR(class_total * 0.85);
        """
    )

    training_rows = connection.execute(
        "SELECT COUNT(*) FROM training_data"
    ).fetchone()[0]

    validation_rows = connection.execute(
        "SELECT COUNT(*) FROM validation_data"
    ).fetchone()[0]

    test_rows = connection.execute(
        "SELECT COUNT(*) FROM test_data"
    ).fetchone()[0]

    print(f"Training rows:   {training_rows:,}")
    print(f"Validation rows: {validation_rows:,}")
    print(f"Test rows:       {test_rows:,}")

    print("\nStep 5/6: Saving the three CSV datasets...")

    connection.execute(
        f"""
        COPY training_data
        TO '{sql_path(TRAIN_FILE)}'
        (HEADER, DELIMITER ',');
        """
    )

    connection.execute(
        f"""
        COPY validation_data
        TO '{sql_path(VALIDATION_FILE)}'
        (HEADER, DELIMITER ',');
        """
    )

    connection.execute(
        f"""
        COPY test_data
        TO '{sql_path(TEST_FILE)}'
        (HEADER, DELIMITER ',');
        """
    )

    print("\nStep 6/6: Saving preparation reports...")

    save_distribution(
        connection,
        "training_data",
        "train_class_distribution.csv",
    )

    save_distribution(
        connection,
        "validation_data",
        "validation_class_distribution.csv",
    )

    save_distribution(
        connection,
        "test_data",
        "test_class_distribution.csv",
    )

    with open(
        REPORT_DIR / "removed_constant_features.csv",
        "w",
        encoding="utf-8",
    ) as report:
        report.write("removed_constant_feature\n")

        for column in constant_columns:
            report.write(f'"{column}"\n')

    summary = {
        "processing_engine": "DuckDB",
        "memory_limit": MEMORY_LIMIT,
        "threads": THREADS,
        "original_rows": int(original_rows),
        "initial_features": len(feature_columns),
        "constant_features_removed": len(constant_columns),
        "remaining_features": len(remaining_features),
        "conflicting_feature_groups_removed": int(
            conflicting_groups
        ),
        "conflicting_rows_removed": int(conflicting_rows),
        "duplicate_rows_removed": int(duplicate_rows_removed),
        "reliable_unique_rows": int(prepared_rows),
        "training_rows": int(training_rows),
        "validation_rows": int(validation_rows),
        "test_rows": int(test_rows),
        "split_method": (
            "Deterministic stratified 70/15/15 split "
            "using feature hashes"
        ),
    }

    with open(
        REPORT_DIR / "preparation_summary.json",
        "w",
        encoding="utf-8",
    ) as summary_file:
        json.dump(summary, summary_file, indent=4)

    connection.close()

    print("=" * 60)
    print("Data preparation completed successfully.")
    print(f"Training dataset:   {TRAIN_FILE}")
    print(f"Validation dataset: {VALIDATION_FILE}")
    print(f"Test dataset:       {TEST_FILE}")
    print(f"Reports directory:  {REPORT_DIR}")


if __name__ == "__main__":
    main()
