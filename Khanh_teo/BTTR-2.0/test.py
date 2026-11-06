import torch
from pytorch_lightning import Trainer

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

test_year = "2014"
ckp_path = "/home/projectg7/workspace/Week6_Recognition/Project Dr.Cuong/pretrained-2014.ckpt"

if __name__ == "__main__":
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(logger=False, accelerator=accelerator, devices=1)

    dm = CROHMEDatamodule(test_year=test_year)

    map_location = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = LitBTTR.load_from_checkpoint(
        ckp_path,
        map_location=map_location,
        weights_only=False,
    )

    trainer.test(model, datamodule=dm)
