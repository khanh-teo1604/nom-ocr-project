import os
import argparse
from ultralytics import YOLO

# GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-d845bc89-502b-9954-b14b-3e51094e9c3a)
# GPU 1: NVIDIA A100-PCIE-40GB (UUID: GPU-5e3f679d-0465-059b-dc09-ab31760ca7dd)

# MIG 3g.20gb     Device  0: (UUID: MIG-b41c5bd0-f3d2-564b-859a-e6f2e060b5ee)
# MIG 2g.10gb     Device  1: (UUID: MIG-e8a45f6e-545a-5aab-ae71-db8d5b336de4)
# MIG 2g.10gb     Device  2: (UUID: MIG-822aef03-bf94-5d72-bd27-dd86770c43e9)

os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-b41c5bd0-f3d2-564b-859a-e6f2e060b5ee,MIG-e8a45f6e-545a-5aab-ae71-db8d5b336de4,MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"

def test(name, model_path, list):
    model = YOLO(model_path)
    metrics = model.val(
    data="/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/datasets/finetune-data.yaml",  # Path to dataset YAML file
    split="test",                # Use 'val' or 'test' depending on your YAML
    imgsz=1280,                 # Match your training resolution
    batch=16,
    device=2                    # GPU ID (or 'cpu')
    )

    list.append([name,
            f"mAP50-95: {metrics.box.map}",
            f"mAP50:    {metrics.box.map50}",
            f"Precision: {metrics.box.mp}",
            f"Recall:    {metrics.box.mr}"])

if __name__ == "__main__":
    final_list = []

    for i in range(16, 17):
        name = f"finetune-{i}"
        model_path = f"/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/runs/{name}/weights/best_recall.pt"
        test(name, model_path, final_list)

    # model_path = "/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/prefinetune_model/best.pt"
    # test("before", model_path, final_list)

    for model_result in final_list:
        print("====================")
        for i in model_result:
            print(i)

