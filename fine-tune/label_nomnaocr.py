from ultralytics import YOLO
import argparse
import os

# GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-d845bc89-502b-9954-b14b-3e51094e9c3a)
# GPU 1: NVIDIA A100-PCIE-40GB (UUID: GPU-5e3f679d-0465-059b-dc09-ab31760ca7dd)
# MIG 3g.20gb     Device  0: (UUID: MIG-b41c5bd0-f3d2-564b-859a-e6f2e060b5ee)
# MIG 2g.10gb     Device  1: (UUID: MIG-e8a45f6e-545a-5aab-ae71-db8d5b336de4)
# MIG 2g.10gb     Device  2: (UUID: MIG-822aef03-bf94-5d72-bd27-dd86770c43e9)

os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"  
subfolders = ["DVSKTT-1 Quyen thu", "DVSKTT-2 Ngoai ky toan thu", "DVSKTT-3 Ban ky toan thu", 
                "DVSKTT-4 Ban ky thuc luc", "DVSKTT-5 Ban ky tuc bien", "Luc Van Tien", "Tale of Kieu 1866",
                "Tale of Kieu 1871", "Tale of Kieu 1872"]

def generate_gt(model_checkpoint, path, output_folder):

    model = YOLO(model_checkpoint)
    model.predict(
        path,
        max_det=500,
        conf=0.1,
        iou=0.7,
        save_txt=True,
        save_conf=True, # might not need?
        batch=1,

        save=True,
        show_labels=False,
        show_conf=False,
        show_boxes=True,
        line_width=1,

        project=output_folder,
        name="",
        imgsz=1280,
        single_cls=True,
        verbose=True,
        device = 0
    ) 


if __name__ == "__main__":
    for subfolder in subfolders:
        parser = argparse.ArgumentParser()
        parser.add_argument("--path", type=str, default=f"/home/projectg7/datasets/nomnaocr/images/{subfolder}/*.jpg")
        parser.add_argument("--model", type=str, default="/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/runs/finetune-6/weights/best.pt")
        parser.add_argument("--output", type=str, default=f"/home/projectg7/workspace/nom-ocr-project/projectg7_finetune/datasets/nomnaocr2/{subfolder}")
        args = parser.parse_args()

        print(f"\n === Labeling {subfolder} === \n")
        generate_gt(args.model, args.path, args.output)

