#!/bin/bash

SRC_BASE="/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/datasets/nomnaocr"
DEST_BASE="/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/datasets/nomnaocr2"

FOLDERS=(
  "DVSKTT-4 Ban ky thuc luc"
  "Tale of Kieu 1866"
  "DVSKTT-2 Ngoai ky toan thu"
  "DVSKTT-5 Ban ky tuc bien"
  "Tale of Kieu 1871"
  "DVSKTT-3 Ban ky toan thu"
  "Luc Van Tien"
  "Tale of Kieu 1872"
  "DVSKTT-1 Quyen thu"
)

for FOLDER in "${FOLDERS[@]}"; do
    SRC_IMAGES_DIR="$SRC_BASE/$FOLDER/images"
    DEST_FOLDER_DIR="$DEST_BASE/$FOLDER"

    if [ -d "$SRC_IMAGES_DIR" ]; then
        echo "Copying images for: $FOLDER"

        # Ensure the destination parent directory exists
        mkdir -p "$DEST_FOLDER_DIR"

        # Copy the images folder recursively into the destination folder
        cp -r "$SRC_IMAGES_DIR" "$DEST_FOLDER_DIR/"

        echo "  -> Successfully copied to $DEST_FOLDER_DIR/images"
    else
        echo "Skipping: $FOLDER (Source images directory not found)"
    fi
done

echo "All copies complete!"