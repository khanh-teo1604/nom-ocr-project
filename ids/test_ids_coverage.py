from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import sys


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Folder containing this Python script
SCRIPT_DIR = Path(__file__).resolve().parent

# Project root: parent folder of ids/
PROJECT_DIR = SCRIPT_DIR.parent

# IDS dictionary
DEFAULT_DICT = SCRIPT_DIR / "ids_final_1.txt"

# Dataset labels directory
DEFAULT_LABELS = (
    PROJECT_DIR
    / "datasets"
    / "tkh-mth2k2"
    / "MTH1000"
    / "label_char"
)

# Output CSV report
DEFAULT_OUTPUT = SCRIPT_DIR / "ids_coverage_report.csv"


# ============================================================
# 2. LOAD IDS DICTIONARY
# ============================================================

def load_ids_dictionary(dict_path):
    """
    Load characters from the IDS dictionary.

    Supported formats:

    Format 1: two columns
        Char<TAB>Fully_Expanded_IDS

    Format 2: four columns
        Code<TAB>Char<TAB>Original_IDS<TAB>Fully_Expanded_IDS

    Returns:
        set: Characters present in the dictionary.
    """

    if not dict_path.is_file():
        raise FileNotFoundError(
            f"IDS dictionary not found: {dict_path}"
        )

    ids_chars = set()
    skipped_lines = 0

    with dict_path.open("r", encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            # Prefer tab-separated parsing because IDS expressions
            # may contain characters that should remain unchanged.
            parts = line.split("\t")

            # Support a four-column dictionary:
            # Code, Char, Original_IDS, Fully_Expanded_IDS
            if len(parts) >= 4 and len(parts[1]) == 1:
                char = parts[1]

            # Support a two-column dictionary:
            # Char, Fully_Expanded_IDS
            elif len(parts) >= 2 and len(parts[0]) == 1:
                char = parts[0]

            else:
                skipped_lines += 1
                continue

            ids_chars.add(char)

    print("=" * 60)
    print("IDS DICTIONARY")
    print("=" * 60)
    print(f"Dictionary path       : {dict_path}")
    print(f"Characters loaded     : {len(ids_chars)}")
    print(f"Skipped lines         : {skipped_lines}")

    if not ids_chars:
        raise ValueError(
            "No characters were loaded from the IDS dictionary. "
            "Please check its format."
        )

    return ids_chars


# ============================================================
# 3. SCAN LABEL FILES
# ============================================================

def check_coverage(label_dir, ids_chars, output_report):
    """
    Recursively scan all TXT files in label_dir.

    Each label line is expected to contain:
        Character x1 y1 x2 y2

    Only the first field (Character) is checked.

    Generates a CSV report containing characters missing
    from the IDS dictionary.
    """

    if not label_dir.is_dir():
        raise NotADirectoryError(
            f"Label directory not found: {label_dir}"
        )

    txt_files = sorted(label_dir.rglob("*.txt"))

    if not txt_files:
        print(f"\nNo TXT files found in: {label_dir}")
        return

    # Count every occurrence of each character.
    char_counter = Counter()

    # Record which files contain each character.
    char_files = defaultdict(set)

    total_lines = 0
    valid_lines = 0
    invalid_lines = 0
    unreadable_files = []

    print("\n" + "=" * 60)
    print("SCANNING LABEL FILES")
    print("=" * 60)
    print(f"Label directory       : {label_dir}")
    print(f"TXT files discovered  : {len(txt_files)}")

    for txt_path in txt_files:
        try:
            with txt_path.open(
                "r",
                encoding="utf-8-sig"
            ) as file:

                for line_number, line in enumerate(file, start=1):
                    total_lines += 1

                    line = line.strip()

                    # Ignore empty lines.
                    if not line:
                        continue

                    # Example:
                    # 紙 309 1116 371 1197
                    parts = line.split()

                    # Expect at least five fields:
                    # Character x1 y1 x2 y2
                    if len(parts) < 5:
                        invalid_lines += 1
                        continue

                    # Only check the first field.
                    char = parts[0]

                    # The dataset is expected to contain one
                    # character per label line.
                    if len(char) != 1:
                        invalid_lines += 1
                        continue

                    valid_lines += 1
                    char_counter[char] += 1

                    # Save the relative path for the report.
                    relative_path = txt_path.relative_to(
                        label_dir
                    ).as_posix()

                    char_files[char].add(relative_path)

        except (OSError, UnicodeError) as error:
            unreadable_files.append(
                (str(txt_path), str(error))
            )

    # ========================================================
    # 4. CALCULATE COVERAGE
    # ========================================================

    dataset_chars = set(char_counter.keys())

    missing_chars = dataset_chars - ids_chars
    covered_chars = dataset_chars & ids_chars

    total_unique_chars = len(dataset_chars)
    total_covered_chars = len(covered_chars)
    total_missing_chars = len(missing_chars)

    total_occurrences = sum(char_counter.values())

    covered_occurrences = sum(
        char_counter[char]
        for char in covered_chars
    )

    missing_occurrences = sum(
        char_counter[char]
        for char in missing_chars
    )

    unique_coverage = (
        total_covered_chars / total_unique_chars * 100
        if total_unique_chars > 0
        else 0.0
    )

    occurrence_coverage = (
        covered_occurrences / total_occurrences * 100
        if total_occurrences > 0
        else 0.0
    )

    # ========================================================
    # 5. WRITE CSV REPORT
    # ========================================================

    output_report.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with output_report.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "Character",
            "Occurrence_Count",
            "Number_of_Files",
            "Files"
        ])

        # List the most frequently occurring missing characters first.
        for char in sorted(
            missing_chars,
            key=lambda c: (-char_counter[c], c)
        ):
            writer.writerow([
                char,
                char_counter[char],
                len(char_files[char]),
                "; ".join(sorted(char_files[char]))
            ])

    # ========================================================
    # 6. PRINT SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("COVERAGE SUMMARY")
    print("=" * 60)

    print(f"TXT files discovered       : {len(txt_files)}")
    print(f"Unreadable files           : {len(unreadable_files)}")
    print(f"Total lines                : {total_lines}")
    print(f"Valid label lines          : {valid_lines}")
    print(f"Invalid/non-label lines    : {invalid_lines}")

    print("-" * 60)

    print(f"Total character occurrences: {total_occurrences}")
    print(f"Unique dataset characters  : {total_unique_chars}")

    print("-" * 60)

    print(f"Characters covered by IDS  : {total_covered_chars}")
    print(f"Characters missing from IDS: {total_missing_chars}")
    print(f"Unique-character coverage  : {unique_coverage:.2f}%")

    print("-" * 60)

    print(f"Covered occurrences        : {covered_occurrences}")
    print(f"Missing occurrences        : {missing_occurrences}")
    print(f"Occurrence coverage        : {occurrence_coverage:.2f}%")

    print("-" * 60)

    print(f"CSV report saved to        : {output_report}")

    # Show the 30 most frequent missing characters.
    if missing_chars:
        print("\nTop 30 missing characters:")
        print("-" * 40)

        for char, count in char_counter.most_common():
            if char in missing_chars:
                print(f"{char} : {count}")

                if sum(
                    1 for c in char_counter.most_common()
                    if c in missing_chars
                    and char_counter[c] >= count
                ) >= 30:
                    # This condition is intentionally not used
                    # to stop output; the loop limit below handles it.
                    pass

                shown = getattr(
                    check_coverage,
                    "_unused",
                    0
                )

        # Print exactly the first 30 missing characters.
        print("\nTop 30 list:")
        for char in sorted(
            missing_chars,
            key=lambda c: (-char_counter[c], c)
        )[:30]:
            print(f"{char} : {char_counter[char]}")

    else:
        print("\nAll dataset characters exist in the IDS dictionary!")

    if unreadable_files:
        print("\nFiles that could not be read:")

        for path, error in unreadable_files:
            print(f"- {path}: {error}")


# ============================================================
# 7. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Check IDS dictionary coverage against "
            "all TXT labels in a dataset directory."
        )
    )

    parser.add_argument(
        "--dict",
        type=Path,
        default=DEFAULT_DICT,
        help="Path to IDS dictionary."
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="Directory containing TXT label files."
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to output CSV report."
    )

    args = parser.parse_args()

    try:
        ids_chars = load_ids_dictionary(args.dict)

        check_coverage(
            args.labels,
            ids_chars,
            args.output
        )

    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()