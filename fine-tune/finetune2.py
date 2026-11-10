import os
import shutil
from ultralytics import YOLO

# GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-d845bc89-502b-9954-b14b-3e51094e9c3a)
# GPU 1: NVIDIA A100-PCIE-40GB (UUID: GPU-5e3f679d-0465-059b-dc09-ab31760ca7dd)

# MIG 3g.20gb     Device  0: (UUID: MIG-b41c5bd0-f3d2-564b-859a-e6f2e060b5ee)
# MIG 2g.10gb     Device  1: (UUID: MIG-e8a45f6e-545a-5aab-ae71-db8d5b336de4)
# MIG 2g.10gb     Device  2: (UUID: MIG-822aef03-bf94-5d72-bd27-dd86770c43e9)

#os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"

best_recall_score = 0.0

def save_best_recall_callback(trainer):
    global best_recall_score
    
    # 1. Extract Bounding Box Mean Recall from the current epoch's validation metrics
    current_recall = trainer.metrics.box.mr
    
    # 2. Compare against the highest recall seen so far
    if current_recall > best_recall_score:
        best_recall_score = current_recall
        
        # Paths to weights directory
        weights_dir = os.path.join(trainer.save_dir, "weights")
        last_ckpt = os.path.join(weights_dir, "last.pt")
        best_recall_ckpt = os.path.join(weights_dir, "best_recall.pt")
        
        # Copy last.pt to best_recall.pt
        if os.path.exists(last_ckpt):
            shutil.copy(last_ckpt, best_recall_ckpt)
            print(f"\n[RECALL CALLBACK] New Best Recall: {best_recall_score:.4f}! Saved to {best_recall_ckpt}")


project = "/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/runs"
name = "finetune"
model = YOLO("/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/runs/finetune-8/weights/best.pt") 

model.add_callback("on_val_end", save_best_recall_callback)

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
