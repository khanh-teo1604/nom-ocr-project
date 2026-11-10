# import glob, os
# from prepare import convert_yolo_single_class

# label_dir = "datasets/tkh-mth2k2/TKH/label_char" & "datasets/tkh-mth2k2/MTH1200/label_char"
# output_dir = "datasets/tkh-mth2k2/TKH/label_char" & "datasets/tkh-mth2k2/MTH1200/label_char"
# os.makedirs(output_dir, exist_ok=True)

# for label_path in glob.glob(f"{label_dir}/*.txt"):
#     output_path = label_path.replace(label_dir, output_dir)
#     convert_yolo_single_class(label_path, output_path)

import glob
import os
from prepare import convert_yolo_single_class

# Define pairs of (input_dir, output_dir)
dir_pairs = [
    ("datasets/tkh-mth2k2/TKH/labels", "datasets/tkh-mth2k2/TKH/labels_single"),
    ("datasets/tkh-mth2k2/MTH1200/labels", "datasets/tkh-mth2k2/MTH1200/labels_single"),
    ("datasets/tkh-mth2k2/MTH1000/labels", "datasets/tkh-mth2k2/MTH1000/labels_single"),
]

for label_dir, output_dir in dir_pairs:
    os.makedirs(output_dir, exist_ok=True)
    
    for label_path in glob.glob(os.path.join(label_dir, "*.txt")):
        filename = os.path.basename(label_path)
        output_path = os.path.join(output_dir, filename)
        
        convert_yolo_single_class(label_path, output_path)