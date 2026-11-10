import os
from ultralytics import YOLO

# GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-d845bc89-502b-9954-b14b-3e51094e9c3a)
# GPU 1: NVIDIA A100-PCIE-40GB (UUID: GPU-5e3f679d-0465-059b-dc09-ab31760ca7dd)

# MIG 3g.20gb     Device  0: (UUID: MIG-b41c5bd0-f3d2-564b-859a-e6f2e060b5ee)
# MIG 2g.10gb     Device  1: (UUID: MIG-e8a45f6e-545a-5aab-ae71-db8d5b336de4)
# MIG 2g.10gb     Device  2: (UUID: MIG-822aef03-bf94-5d72-bd27-dd86770c43e9)

#os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"
project = "/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/runs"
name = "finetune"
model = YOLO("/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/runs/finetune-8/weights/best.pt") 

model.train(data="datasets/finetune-data.yaml",
            epochs=100,
            imgsz=1280,
            single_cls=True,
            batch=32,
            patience=10,
            save=True,
            # save_period=1,
            project=project,
            name=name,

            mosaic=0,
            resume=False,
            # device=0,
            # cache=True,
            plots=True,
            # crop_fraction=0.5,
            pretrained=True,
            deterministic=False,
            )
