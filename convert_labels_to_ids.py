#!/usr/bin/env python3
"""
convert_labels_to_ids.py

Converts character crop labels in tkh-mth2k2-crops to Ideographic Description Sequences (IDS)
using IDSMapper.char_to_ids(char), and updates metadata.csv with both original character and IDS.
"""

import csv
import os
import sys
from pathlib import Path
from tqdm import tqdm

# Add phuc directory to path for IDSMapper
sys.path.append(str(Path(__file__).parent / "phuc"))
from ids_mapper import IDSMapper


def convert_subset(subset_name: str, base_dir: Path, mapper: IDSMapper):
    subset_dir = base_dir / subset_name
    labels_file = subset_dir / "labels.txt"
    metadata_file = subset_dir / "metadata.csv"

    if not labels_file.exists() or not metadata_file.exists():
        print(f"Skipping {subset_name}: files not found.")
        return

    print(f"\n==================================================")
    print(f"Processing subset: {subset_name}")
    print(f"==================================================")

    # 1. Update metadata.csv
    print(f"Updating {metadata_file} ...")
    temp_metadata = subset_dir / "metadata.csv.tmp"
    
    total_rows = 0
    decomposed_count = 0
    atomic_count = 0

    with open(metadata_file, "r", encoding="utf-8") as f_in, \
         open(temp_metadata, "w", encoding="utf-8", newline="") as f_out:
        
        reader = csv.DictReader(f_in)
        fieldnames = ["crop_path", "char", "ids", "page_id", "box_idx", "x1", "y1", "x2", "y2", "crop_w", "crop_h"]
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in tqdm(reader, desc=f"Metadata {subset_name}"):
            char = row["char"]
            ids_seq = mapper.char_to_ids(char)
            row["ids"] = ids_seq
            
            # Select/reorder fields
            ordered_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(ordered_row)

            total_rows += 1
            if ids_seq != char:
                decomposed_count += 1
            else:
                atomic_count += 1

    os.replace(temp_metadata, metadata_file)
    print(f"  -> Saved {metadata_file} ({total_rows:,} rows)")

    # 2. Update labels.txt
    print(f"Updating {labels_file} ...")
    temp_labels = subset_dir / "labels.txt.tmp"

    with open(labels_file, "r", encoding="utf-8") as f_in, \
         open(temp_labels, "w", encoding="utf-8") as f_out:
        
        for line in tqdm(f_in, desc=f"Labels {subset_name}", total=total_rows):
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                crop_path, char = parts[0], parts[1]
                ids_seq = mapper.char_to_ids(char)
                f_out.write(f"{crop_path}\t{ids_seq}\n")
            elif len(parts) == 1 and parts[0]:
                f_out.write(f"{line.strip()}\n")

    os.replace(temp_labels, labels_file)
    print(f"  -> Saved {labels_file} ({total_rows:,} lines)")

    # Summary
    print(f"[{subset_name}] Complete:")
    print(f"  - Total processed: {total_rows:,}")
    print(f"  - Decomposed into multi-part IDS: {decomposed_count:,} ({decomposed_count / max(1, total_rows) * 100:.2f}%)")
    print(f"  - Atomic/Single characters: {atomic_count:,} ({atomic_count / max(1, total_rows) * 100:.2f}%)")


def main():
    base_dir = Path("/home/projectg7/workspace/nom-ocr-project/datasets/tkh-mth2k2-crops")
    subsets = ["MTH1000", "MTH1200", "TKH"]

    print("Initializing IDSMapper...")
    mapper = IDSMapper()

    for subset in subsets:
        convert_subset(subset, base_dir, mapper)

    print("\n✅ All subsets updated to IDS successfully!")


if __name__ == "__main__":
    main()

