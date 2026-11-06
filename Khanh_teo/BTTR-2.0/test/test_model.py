import os
import warnings
from pathlib import Path

# Suppress harmless migration warnings from PyTorch Lightning
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")

os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-822aef03-bf94-5d72-bd27-dd86770c43e9"

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from bttr.lit_bttr import LitBTTR

# Resolve the exact checkpoint inside your local 'Project Dr.Cuong' folder
PROJECT_DIR = Path(__file__).resolve().parents[1].parent
ckpt = PROJECT_DIR / "pretrained-2014.ckpt"
img_path = PROJECT_DIR / "/home/projectg7/workspace/Week6_Recognition/Project Dr.Cuong/BTTR-2.0/2016/UN_101_em_8.bmp"

print(f"Loading checkpoint directly from your folder: {ckpt.resolve()}")

# Load model (with map_location and weights_only=False)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = LitBTTR.load_from_checkpoint(str(ckpt), map_location=device, weights_only=False)
model.eval()

# Process image
img = Image.open(str(img_path)).convert("L")
img_tensor = ToTensor()(img)
if torch.cuda.is_available():
    model = model.cuda()
    img_tensor = img_tensor.cuda()

# Beam search prediction
hyp = model.beam_search(img_tensor)
print("Predicted LaTeX:", hyp)