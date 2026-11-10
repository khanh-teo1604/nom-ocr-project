from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionPredictor
import os


# MIG 3g.20gb     Device  0: (UUID: MIG-b41c5bd0-f3d2-564b-859a-e6f2e060b5ee)
# MIG 2g.10gb     Device  1: (UUID: MIG-e8a45f6e-545a-5aab-ae71-db8d5b336de4)
# MIG 2g.10gb     Device  2: (UUID: MIG-822aef03-bf94-5d72-bd27-dd86770c43e9)
os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"  # specify which GPU to use

images = ["DVSKTT_ban_thuc_XI_4a_jpg.rf.6fa0a2d06bd2869800959cd9cbc47519.jpg",
            "DVSKTT_ban_thuc_XII_47b_jpg.rf.18c34e248bc960cc3f2a239d2286924f.jpg",
            "nlvnpf-0059-069_jpg.rf.4d40e37b2ddc5ef75d7ff10ff694519f.jpg",
            "page046_jpg.rf.169e709e9ed7d712ba610694842d7314.jpg",
            "DVSKTT_ngoai_IV_20a_jpg.rf.c9baa74de71ea001147a07fc810023ee.jpg",
            "DVSKTT_ban_tuc_XVII_53b_jpg.rf.67a6bb4e370470b81a48df3ce20466ff.jpg",
            "DVSKTT_ban_toan_X_24b_jpg.rf.efcccd7bb502bd1c7ff28ab1209b63df.jpg",
            "page016_jpg.rf.da27495b3fdfcfc6fadd2116d7ffacc0.jpg",
            "page76a_jpg.rf.0eee956ea0a735ea06927b2f286ac517.jpg",
            "DVSKTT_ngoai_V_26b_jpg.rf.809d435f981760ec3be1d3c744b7a047.jpg",
            "page050_jpg.rf.2136a024971e80e1144b7c16ad6feb9d.jpg"
            ]


if __name__ == "__main__":
    model = YOLO("/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/runs/finetune-8/weights/best.pt")
    
    for image in images:
        model.predict(
            f"/home/projectg7/datasets/nom-detection-testset/images",
            save=True,
            conf=0.1,
            iou=0.7,
            max_det=500,
            line_width=1,
            show_labels=False,
            show_conf=False,
            project="/home/projectg7/workspace/nom-ocr-project/projectg7_finetune",
            name="predict(finetune-8-0.1-full)",
            imgsz=1280,
            device=0
        )


