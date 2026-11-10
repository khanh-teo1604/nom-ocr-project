# from ultralytics import YOLO

# model = YOLO(
#     "runs/detect/lab-runs/tkh-mth2k2-train-3/weights/best.pt")      #train from best.pt
# model.train(
#     data="/home/projectg7/workspace/nom-ocr-project/trac/datasets/tkh-mth2k2/data.yaml",
#     epochs=50,
#     imgsz=1280,
#     single_cls=True,             # collapse to one class even if labels weren't pre-converted
#     batch=4,
#     patience=10,
#     project="lab-runs",
#     name="tkh-mth2k2-train",
# )
from ultralytics import YOLO

model = YOLO("/home/projectg7/workspace/nom-ocr-project/trac/runs/detect/lab-runs/tkh-mth2k2-yolo11s-1280/weights/best.pt")

model.train(
    data="/home/projectg7/workspace/nom-ocr-project/trac/datasets/tkh-mth2k2/data.yaml",
    epochs=100,
    imgsz=1280,
    single_cls=True,
    batch=4,
    patience=20,
    save=True,
    save_period=1,
    project="/home/projectg7/workspace/nom-ocr-project/trac/runs/detect/lab-runs",
    name="tkh-mth2k2-yolo11s-1280",
)