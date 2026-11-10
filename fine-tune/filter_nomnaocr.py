import os
import glob
import numpy as np
from PIL import Image
from pathlib import Path

subfolders = ["DVSKTT-1 Quyen thu", "DVSKTT-2 Ngoai ky toan thu", "DVSKTT-3 Ban ky toan thu", 
                "DVSKTT-4 Ban ky thuc luc", "DVSKTT-5 Ban ky tuc bien", "Luc Van Tien", "Tale of Kieu 1866",
                "Tale of Kieu 1871", "Tale of Kieu 1872"]

def filter_function(image_path, model_label_path, line_label_path, output_path):
    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = []

    with open(model_label_path) as model_label_file:
        for model_label in model_label_file:
            x_center, y_center, width, height = map(float, model_label.split()[1:5])
            x_center_real, y_center_real = (x_center * img_width, y_center * img_height)

            with open(line_label_path) as line_label_file:
                for line_label in line_label_file:
                    x1, y1, x2, y2, x3, y3, x4, y4 = map(float, line_label.split(",")[:8])

                    within_lower_bound = (y_center_real > y1 and y_center_real > y2)
                    within_upper_bound = (y_center_real < y3 and y_center_real < y4)
                    within_left_bound = (x_center_real > x1 and x_center_real > x4)
                    within_right_bound  = (x_center_real < x2 and x_center_real < x3)

                    if within_lower_bound and within_upper_bound and within_right_bound and within_left_bound:
                        annotations.append((x_center, y_center, width, height)) 
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        for i in range(len(annotations)):
            f.write(f'0 {annotations[i][0]:6f} {annotations[i][1]:6f} {annotations[i][2]:6f} {annotations[i][3]:6f}\n')

# if __name__ == "__main__": 
#     for subfolder in subfolders:
#         line_label_path = f"/home/projectg7/datasets/nomnaocr/line_labels/{subfolder}/*.txt"
#         label_path = f"/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/datasets/nomnaocr/{subfolder}/labels/*.txt"
#         real_label_path = f"/home/projectg7/datasets/nomnaocr/detection_labels/{subfolder}/*.txt"
#         output_path = f"/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/datasets/nomnaocr/{subfolder}/filtered_labels/*.txt" 
#         filter_function()

if __name__ == "__main__": 
    base_img_dir = "/home/projectg7/datasets/nomnaocr/images"
    base_line_dir = "/home/projectg7/datasets/nomnaocr/line_labels"
    base_pred_dir = "/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/datasets/nomnaocr2"

    for subfolder in subfolders:
        pred_label_dir = os.path.join(base_pred_dir, subfolder, "prefilter_labels")
        line_label_dir = os.path.join(base_line_dir, subfolder)
        image_dir = os.path.join(base_img_dir, subfolder)
        output_dir = os.path.join(base_pred_dir, subfolder, "labels")

        # Get all text files in predicted labels directory
        pred_files = glob.glob(os.path.join(pred_label_dir, "*.txt"))
        
        print(f"Processing subfolder '{subfolder}': found {len(pred_files)} files.")

        for model_label_path in pred_files:
            file_stem = Path(model_label_path).stem  # e.g., "page_001"
            
            # Find matching image file (.jpg or .png)
            image_path = os.path.join(image_dir, f"{file_stem}.jpg")
            if not os.path.exists(image_path):
                image_path = os.path.join(image_dir, f"{file_stem}.png")

            line_label_path = os.path.join(line_label_dir, f"{file_stem}.txt")
            output_path = os.path.join(output_dir, f"{file_stem}.txt")

            if os.path.exists(image_path):
                filter_function(image_path, model_label_path, line_label_path, output_path)
            else:
                print(f"Warning: Image file not found for {file_stem}")