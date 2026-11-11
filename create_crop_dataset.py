#!/usr/bin/env python3
"""
create_crop_dataset.py

Crops character bounding boxes from full-page document images in tkh-mth2k2
(MTH1000, MTH1200, TKH) and organizes them into an OCR-ready dataset
with labels.txt and metadata.csv.
"""

import argparse
import csv
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


def sanitize_filename_char(char: str) -> str:
    """Sanitize characters that are invalid in filenames."""
    return re.sub(r'[\\/*?:"<>|]', "_", char)


def process_single_page(
    args: Tuple[str, str, str, str]
) -> Tuple[str, List[Tuple[str, str]], List[dict], int]:
    """
    Process a single page image and its corresponding label_char file.

    Parameters
    ----------
    args : Tuple[img_path, label_path, images_out_dir, subset_name]

    Returns
    -------
    Tuple[stem, label_records, metadata_records, skipped_count]
    """
    img_path, label_path, images_out_dir, subset_name = args
    stem = Path(img_path).stem

    if not os.path.exists(label_path):
        return stem, [], [], 0

    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_w, img_h = img.size

            label_records: List[Tuple[str, str]] = []
            metadata_records: List[dict] = []
            skipped_count = 0

            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for box_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    skipped_count += 1
                    continue

                char = parts[0]
                try:
                    x1 = int(round(float(parts[1])))
                    y1 = int(round(float(parts[2])))
                    x2 = int(round(float(parts[3])))
                    y2 = int(round(float(parts[4])))
                except ValueError:
                    skipped_count += 1
                    continue

                # Ensure canonical coordinates (min <= max)
                x_min, x_max = min(x1, x2), max(x1, x2)
                y_min, y_max = min(y1, y2), max(y1, y2)

                # Clamp to image boundaries
                x_min = max(0, min(x_min, img_w))
                x_max = max(0, min(x_max, img_w))
                y_min = max(0, min(y_min, img_h))
                y_max = max(0, min(y_max, img_h))

                # Check for degenerate or zero-area box
                if (x_max - x_min) <= 0 or (y_max - y_min) <= 0:
                    skipped_count += 1
                    continue

                # Crop character
                crop = img.crop((x_min, y_min, x_max, y_max))

                safe_char = sanitize_filename_char(char)
                crop_filename = f"{stem}_{box_idx:04d}_{safe_char}.png"
                crop_full_path = os.path.join(images_out_dir, crop_filename)

                crop.save(crop_full_path)

                # Relative path for labels.txt: images/<filename>
                rel_img_path = f"images/{crop_filename}"
                label_records.append((rel_img_path, char))

                metadata_records.append(
                    {
                        "crop_path": rel_img_path,
                        "char": char,
                        "page_id": stem,
                        "box_idx": box_idx,
                        "x1": x_min,
                        "y1": y_min,
                        "x2": x_max,
                        "y2": y_max,
                        "crop_w": x_max - x_min,
                        "crop_h": y_max - y_min,
                    }
                )

            return stem, label_records, metadata_records, skipped_count

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return stem, [], [], 1


def process_subset(
    subset_name: str,
    src_dir: str,
    out_dir: str,
    num_workers: int,
    limit_pages: Optional[int] = None,
):
    """
    Process all pages in a given subset (e.g. MTH1000).
    """
    subset_src = os.path.join(src_dir, subset_name)
    images_dir = os.path.join(subset_src, "images")
    label_char_dir = os.path.join(subset_src, "label_char")

    if not os.path.exists(images_dir) or not os.path.exists(label_char_dir):
        print(f"Skipping {subset_name}: directory {images_dir} or {label_char_dir} not found.")
        return

    # Output paths
    subset_out = os.path.join(out_dir, subset_name)
    crop_images_out = os.path.join(subset_out, "images")
    labels_file = os.path.join(subset_out, "labels.txt")
    metadata_file = os.path.join(subset_out, "metadata.csv")

    os.makedirs(crop_images_out, exist_ok=True)

    # Collect images
    image_paths = sorted(
        glob.glob(os.path.join(images_dir, "*.png"))
        + glob.glob(os.path.join(images_dir, "*.jpg"))
        + glob.glob(os.path.join(images_dir, "*.jpeg"))
        + glob.glob(os.path.join(images_dir, "*.bmp"))
    )

    if limit_pages is not None:
        image_paths = image_paths[:limit_pages]

    total_pages = len(image_paths)
    print(f"\n[{subset_name}] Processing {total_pages} page(s) with {num_workers} workers...")

    tasks = []
    for img_p in image_paths:
        stem = Path(img_p).stem
        lbl_p = os.path.join(label_char_dir, f"{stem}.txt")
        tasks.append((img_p, lbl_p, crop_images_out, subset_name))

    all_labels: List[Tuple[str, str]] = []
    all_metadata: List[dict] = []
    total_skipped = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_page, task): task[0] for task in tasks}

        for future in tqdm(as_completed(futures), total=total_pages, desc=f"Cropping {subset_name}"):
            stem, label_recs, meta_recs, skipped = future.result()
            all_labels.extend(label_recs)
            all_metadata.extend(meta_recs)
            total_skipped += skipped

    # Sort labels and metadata for consistency
    all_labels.sort(key=lambda x: x[0])
    all_metadata.sort(key=lambda x: (x["page_id"], x["box_idx"]))

    # Write labels.txt
    with open(labels_file, "w", encoding="utf-8") as f:
        for rel_path, char in all_labels:
            f.write(f"{rel_path}\t{char}\n")

    # Write metadata.csv
    fieldnames = ["crop_path", "char", "page_id", "box_idx", "x1", "y1", "x2", "y2", "crop_w", "crop_h"]
    with open(metadata_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metadata)

    print(f"[{subset_name}] Complete:")
    print(f"  - Pages processed: {total_pages}")
    print(f"  - Total character crops: {len(all_labels)}")
    print(f"  - Skipped/degenerate boxes: {total_skipped}")
    print(f"  - Labels saved: {labels_file}")
    print(f"  - Metadata saved: {metadata_file}")


def main():
    parser = argparse.ArgumentParser(description="Crop characters from tkh-mth2k2 dataset.")
    parser.add_argument(
        "--src_dir",
        type=str,
        default="/home/projectg7/datasets/tkh-mth2k2",
        help="Path to tkh-mth2k2 root directory.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/projectg7/workspace/nom-ocr-project/datasets/tkh-mth2k2-crops",
        help="Path to output cropped dataset.",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["MTH1000", "MTH1200", "TKH"],
        help="Subsets to process (default: MTH1000 MTH1200 TKH).",
    )
    parser.add_argument(
        "--limit_pages",
        type=int,
        default=None,
        help="Limit number of pages to process per subset (useful for testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(32, os.cpu_count() or 4),
        help="Number of parallel processes.",
    )

    args = parser.parse_args()

    print("==================================================")
    print("TKH-MTH2k2 Character Cropping Tool")
    print(f"Source Dir  : {args.src_dir}")
    print(f"Output Dir  : {args.out_dir}")
    print(f"Subsets     : {args.subsets}")
    print(f"Limit Pages : {args.limit_pages if args.limit_pages is not None else 'All'}")
    print(f"Workers     : {args.workers}")
    print("==================================================")

    for subset in args.subsets:
        process_subset(
            subset_name=subset,
            src_dir=args.src_dir,
            out_dir=args.out_dir,
            num_workers=args.workers,
            limit_pages=args.limit_pages,
        )


if __name__ == "__main__":
    main()

