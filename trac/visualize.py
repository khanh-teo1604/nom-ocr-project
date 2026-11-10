from pathlib import Path
from ultralytics import YOLO

model = YOLO(
    "/home/projectg7/workspace/nom-ocr-project/trac/"
    "runs/detect/lab-runs/tkh-mth2k2-yolo11s-1280/weights/best.pt"
)

image_dir = Path(
    "/home/projectg7/datasets/nomnaocr/images/"
    "Luc Van Tien"
)

image_paths = sorted(
    path for path in image_dir.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)[:100]

model.predict(
    source=[str(path) for path in image_paths],
    conf=0.25,
    imgsz=640,
    batch=1,
    save=True,
    show_labels=False,
    show_conf=False,
    line_width=1,
    project="/home/projectg7/workspace/nom-ocr-project/trac/runs/detect/lab-runs",
    name="predict-first-100-nomnaocr-Luc Van Tien",
    exist_ok=True,
)