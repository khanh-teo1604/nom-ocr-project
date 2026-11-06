import argparse
import os
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from bttr.datamodule import CROHMEDatamodule, NomCropDataModule, default_ids_dict
from bttr.lit_bttr import LitBTTR


def main():
    parser = argparse.ArgumentParser(description="Train BTTR Model (PyTorch Lightning 2.x compatible)")

    # Dataset selection
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="nom_crop",
        choices=["nom_crop", "crohme"],
        help="Dataset type to train on (nom_crop or crohme)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/projectg7/workspace/nom-ocr-project/datasets/tkh-mth2k2-crops/MTH1000",
        help="Path to cropped dataset directory (for nom_crop)",
    )
    parser.add_argument(
        "--dict_path",
        type=str,
        default=default_ids_dict(),
        help="Path to IDS token dictionary file",
    )
    parser.add_argument(
        "--zipfile_path",
        type=str,
        default="data.zip",
        help="Path to CROHME data.zip (for crohme)",
    )
    parser.add_argument(
        "--test_year",
        type=str,
        default="2014",
        help="CROHME validation year",
    )

    # Training args
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--max_epochs", type=int, default=300, help="Max training epochs")
    parser.add_argument(
        "--check_val_every_n_epoch",
        type=int,
        default=1,
        help="Validation interval (default: 1 epoch)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--learning_rate", type=float, default=1.0, help="Adadelta learning rate")
    parser.add_argument("--patience", type=int, default=20, help="LR scheduler patience")
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Early stopping patience (stop if no improvement after N epochs)",
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=5,
        help="Save top K epoch checkpoints based on val_ExpRate",
    )
    parser.add_argument(
        "--limit_val_batches",
        type=float,
        default=1000,
        help="Number or fraction of validation batches to evaluate per epoch (e.g. 1000 or 1.0 for full)",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="Run 1 train and 1 val batch for sanity testing",
    )

    # Model architecture args
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--growth_rate", type=int, default=24)
    parser.add_argument("--num_layers", type=int, default=16)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_decoder_layers", type=int, default=3)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--beam_size", type=int, default=10)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=1.0)

    # Checkpoints
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--pretrained_ckpt", type=str, default=None, help="Checkpoint for fine-tuning")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Checkpoint to resume full training state")

    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    # 1. Initialize DataModule
    if args.dataset_type == "nom_crop":
        print(f"Initializing NomCropDataModule from {args.data_dir}...")
        dm = NomCropDataModule(
            data_dir=args.data_dir,
            dict_path=args.dict_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_ratio=0.8,
            seed=args.seed,
        )
        vocab_size = len(dm.vocab)
        vocab = dm.vocab
    else:
        print(f"Initializing CROHMEDatamodule from {args.zipfile_path}...")
        dm = CROHMEDatamodule(
            zipfile_path=args.zipfile_path,
            test_year=args.test_year,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        vocab_size = len(dm.vocab) if hasattr(dm, "vocab") else None
        vocab = getattr(dm, "vocab", None)

    # 2. Initialize or Load Model
    if args.pretrained_ckpt:
        print(f"Loading model weights from {args.pretrained_ckpt} for fine-tuning...")
        model = LitBTTR.load_from_checkpoint(
            args.pretrained_ckpt,
            weights_only=False,
            learning_rate=args.learning_rate,
            patience=args.patience,
            vocab_size=vocab_size,
            vocab=vocab,
        )
    else:
        print(f"Initializing model from scratch (vocab_size: {vocab_size})...")
        model = LitBTTR(
            d_model=args.d_model,
            growth_rate=args.growth_rate,
            num_layers=args.num_layers,
            nhead=args.nhead,
            num_decoder_layers=args.num_decoder_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            beam_size=args.beam_size,
            max_len=args.max_len,
            alpha=args.alpha,
            learning_rate=args.learning_rate,
            patience=args.patience,
            vocab_size=vocab_size,
            vocab=vocab,
        )

    # 3. Callbacks: Top-5 checkpoints + EarlyStopping + LRMonitor
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.checkpoint_dir,
        filename="epoch={epoch:02d}-step={step}-val_ExpRate={val_ExpRate:.4f}",
        monitor="val_ExpRate",
        mode="max",
        save_top_k=args.save_top_k,
        save_last=True,
    )
    early_stopping_cb = EarlyStopping(
        monitor="val_ExpRate",
        mode="max",
        patience=args.early_stopping_patience,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # 4. Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    val_batches = int(args.limit_val_batches) if args.limit_val_batches > 1 else args.limit_val_batches

    trainer = Trainer(
        accelerator=accelerator,
        devices=args.gpus if torch.cuda.is_available() else "auto",
        strategy="ddp" if (args.gpus > 1 and torch.cuda.is_available()) else "auto",
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=[checkpoint_cb, early_stopping_cb, lr_monitor],
        limit_val_batches=val_batches,
        fast_dev_run=args.fast_dev_run,
    )

    # 5. Start Training
    print("Starting training...")
    trainer.fit(model, datamodule=dm, ckpt_path=args.resume_ckpt)


if __name__ == "__main__":
    main()
