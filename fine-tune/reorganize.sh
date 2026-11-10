#!/bin/bash

BASE_PATH="/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/datasets/nomnaocr2"

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
    PREDICT_DIR="$BASE_PATH/$FOLDER/predict"
    LABELED_IMAGES_DIR="$BASE_PATH/$FOLDER/labeled_images"
    PREFILTER_LABELS_DIR="$BASE_PATH/$FOLDER/prefilter_labels"

    # Check if the predict directory exists
    if [ -d "$PREDICT_DIR" ]; then
        echo "Processing: $FOLDER"

        # Create the new target directories
        mkdir -p "$LABELED_IMAGES_DIR"
        mkdir -p "$PREFILTER_LABELS_DIR"

        # 1. Handle the labels subfolder
        if [ -d "$PREDICT_DIR/labels" ]; then
            # Move all label files to prefilter_labels
            mv "$PREDICT_DIR/labels/"* "$PREFILTER_LABELS_DIR/" 
            
            # Remove the old labels directory
            rmdir "$PREDICT_DIR/labels" 
        fi

        # 2. Handle the images
        # Move all remaining files in predict/ (which should just be the images now) to labeled_images
        mv "$PREDICT_DIR/"* "$LABELED_IMAGES_DIR/" 

        # 3. Cleanup
        # Remove the now-empty predict directory
        rmdir "$PREDICT_DIR" 

        echo "  -> Successfully reorganized into labeled_images/ and prefilter_labels/"
    else
        echo "Skipping: $FOLDER (No 'predict' directory found)"
    fi
done

echo "Reorganization complete!"