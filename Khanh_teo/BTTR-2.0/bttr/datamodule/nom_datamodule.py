import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms

from .datamodule import Batch
from .vocab import CROHMEVocab, default_ids_dict


class NomCropDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, str]], vocab: CROHMEVocab):
        """
        samples: List of (img_base, full_img_path, ids_seq)
        """
        self.samples = samples
        self.vocab = vocab
        self.transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor, List[int]]:
        img_base, img_path, ids_seq = self.samples[idx]

        with Image.open(img_path) as img:
            img = img.convert("L")  # Grayscale: [1, H, W]
            img_tensor = self.transform(img)

        tokens = list(ids_seq)
        indices = self.vocab.words2indices(tokens)

        return img_base, img_tensor, indices


def nom_collate_fn(batch: List[Tuple[str, torch.Tensor, List[int]]]) -> Batch:
    img_bases = [b[0] for b in batch]
    imgs = [b[1] for b in batch]
    indices = [b[2] for b in batch]

    heights = [img.shape[1] for img in imgs]
    widths = [img.shape[2] for img in imgs]
    max_h = max(heights)
    max_w = max(widths)

    b_size = len(imgs)
    x = torch.zeros(b_size, 1, max_h, max_w, dtype=torch.float32)
    x_mask = torch.ones(b_size, max_h, max_w, dtype=torch.bool)

    for i, img in enumerate(imgs):
        h, w = img.shape[1], img.shape[2]
        x[i, :, :h, :w] = img
        x_mask[i, :h, :w] = 0

    return Batch(img_bases=img_bases, imgs=x, mask=x_mask, indices=indices)


class NomCropDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/home/projectg7/workspace/nom-ocr-project/datasets/tkh-mth2k2-crops/MTH1000",
        dict_path: str = default_ids_dict(),
        batch_size: int = 16,
        num_workers: int = 4,
        train_ratio: float = 0.8,
        seed: int = 7,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.dict_path = dict_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ratio = train_ratio
        self.seed = seed
        self.vocab = CROHMEVocab(dict_path=self.dict_path)

    def setup(self, stage: Optional[str] = None) -> None:
        labels_file = self.data_dir / "labels.txt"
        if not labels_file.exists():
            raise FileNotFoundError(f"labels.txt not found in {self.data_dir}")

        samples: List[Tuple[str, str, str]] = []
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    rel_path, ids_seq = parts[0], parts[1]
                    full_path = str(self.data_dir / rel_path)
                    img_base = Path(rel_path).stem
                    samples.append((img_base, full_path, ids_seq))

        # Shuffle with fixed seed for reproducibility
        rng = random.Random(self.seed)
        rng.shuffle(samples)

        n_train = int(len(samples) * self.train_ratio)
        train_samples = samples[:n_train]
        val_samples = samples[n_train:]

        print(f"[NomCropDataModule] Total samples: {len(samples):,}")
        print(f"  - Train: {len(train_samples):,} ({self.train_ratio*100:.0f}%)")
        print(f"  - Val:   {len(val_samples):,} ({(1-self.train_ratio)*100:.0f}%)")
        print(f"  - Vocab size: {len(self.vocab):,} tokens")

        self.train_dataset = NomCropDataset(train_samples, self.vocab)
        self.val_dataset = NomCropDataset(val_samples, self.vocab)
        self.test_dataset = NomCropDataset(val_samples, self.vocab)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=nom_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=nom_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=nom_collate_fn,
            pin_memory=torch.cuda.is_available(),
        )

