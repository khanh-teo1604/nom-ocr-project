import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import editdistance
import torch
from PIL import Image
from torchvision.transforms import transforms

# Ensure bttr can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bttr.datamodule import CROHMEVocab, default_ids_dict
from bttr.lit_bttr import LitBTTR


SCRIPT_DIR = Path(__file__).resolve().parent

def parse_args():
    parser = argparse.ArgumentParser(description="Test BTTR checkpoint on MTH1200 sample")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=str(SCRIPT_DIR / "checkpoints2/epoch=epoch=11-step=step=504660-val_ExpRate=val_ExpRate=0.9120.ckpt"),
        help="Path to checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/projectg7/workspace/nom-ocr-project/datasets/tkh-mth2k2-crops/MTH1200",
        help="Path to MTH1200 cropped dataset directory",
    )
    parser.add_argument(
        "--dict_path",
        type=str,
        default=default_ids_dict(),
        help="Path to dictionary_ids.txt",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sample selection",
    )
    parser.add_argument(
        "--artifact_dir",
        type=str,
        default="/home/projectg7/.gemini/antigravity/brain/6e28f53b-d95c-454a-9d5c-0e0fa9164fe2",
        help="Path to artifacts directory to save results and images",
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        default="mth1200_evaluation_report.pdf",
        help="Path where PDF report will be saved",
    )
    return parser.parse_args()


def load_mth1200_samples(data_dir: Path, num_samples: int, seed: int) -> List[Tuple[str, str, str]]:
    labels_file = data_dir / "labels.txt"
    if not labels_file.exists():
        raise FileNotFoundError(f"labels.txt not found in {data_dir}")

    all_samples: List[Tuple[str, str, str]] = []
    print(f"Reading labels from {labels_file}...")
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                rel_path, ids_seq = parts[0], parts[1]
                full_img_path = str(data_dir / rel_path)
                img_name = Path(rel_path).name
                all_samples.append((img_name, full_img_path, ids_seq))

    print(f"Total samples loaded from labels.txt: {len(all_samples):,}")

    rng = random.Random(seed)
    # Check image file existence when sampling
    rng.shuffle(all_samples)
    
    valid_samples = []
    for sample in all_samples:
        if os.path.exists(sample[1]):
            valid_samples.append(sample)
            if len(valid_samples) == num_samples:
                break

    if len(valid_samples) < num_samples:
        raise RuntimeError(f"Could only find {len(valid_samples)} valid image files out of {num_samples} requested.")

    return valid_samples


def main():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Vocab
    print(f"Loading vocabulary from {args.dict_path}...")
    vocab = CROHMEVocab(dict_path=args.dict_path)
    print(f"Vocab size: {len(vocab):,}")

    # 2. Load Checkpoint
    ckpt_path = os.path.abspath(args.ckpt)
    print(f"Loading checkpoint: {ckpt_path}...")
    model = LitBTTR.load_from_checkpoint(
        ckpt_path,
        weights_only=False,
        vocab=vocab,
        vocab_size=len(vocab),
        map_location=device,
    )
    model.eval()
    model.to(device)

    # 3. Load Samples
    data_dir = Path(args.data_dir)
    samples = load_mth1200_samples(data_dir, args.num_samples, args.seed)

    # 4. Prepare Output Artifact Folder
    artifact_dir = Path(args.artifact_dir)
    images_artifact_dir = artifact_dir / "sample_images"
    images_artifact_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.ToTensor()

    print("\nRunning inference on 10 samples...")
    results = []

    for idx, (img_name, full_img_path, gt_ids) in enumerate(samples, 1):
        # Copy image to artifact directory for embedding
        dest_img_path = images_artifact_dir / img_name
        shutil.copy2(full_img_path, dest_img_path)

        with Image.open(full_img_path) as img:
            img_gray = img.convert("L")
            img_w, img_h = img_gray.size
            img_tensor = transform(img_gray).unsqueeze(0).to(device)  # [1, 1, H, W]

        mask = torch.zeros((1, img_tensor.shape[2], img_tensor.shape[3]), dtype=torch.bool, device=device)

        with torch.no_grad():
            hyps = model.bttr.beam_search(
                img_tensor, mask, beam_size=model.hparams.beam_size, max_len=model.hparams.max_len
            )
            best_hyp = max(hyps, key=lambda h: h.score / (len(h) ** model.hparams.alpha))

        # Filter out special tokens if any, and decode
        clean_seq = [token_id for token_id in best_hyp.seq if token_id not in (vocab.PAD_IDX, vocab.SOS_IDX, vocab.EOS_IDX)]
        pred_ids = vocab.indices2label(clean_seq, separator="")

        is_match = (pred_ids == gt_ids)
        dist = editdistance.eval(list(pred_ids), list(gt_ids))

        results.append({
            "idx": idx,
            "filename": img_name,
            "size": f"{img_w}x{img_h}",
            "artifact_img_path": str(dest_img_path),
            "gt_ids": gt_ids,
            "pred_ids": pred_ids,
            "match": is_match,
            "edit_dist": dist,
            "score": float(best_hyp.score),
        })

    # Summary
    total = len(results)
    matches = sum(1 for r in results if r["match"])
    accuracy = (matches / total) * 100
    mean_dist = sum(r["edit_dist"] for r in results) / total

    print("\n" + "=" * 90)
    print(f"{'Idx':<4} | {'Image Name':<30} | {'Ground Truth IDS':<18} | {'Predicted IDS':<18} | {'Status':<6}")
    print("-" * 90)
    for r in results:
        status = "MATCH" if r["match"] else "DIFF"
        print(f"{r['idx']:<4} | {r['filename']:<30} | {r['gt_ids']:<18} | {r['pred_ids']:<18} | {status:<6}")
    print("=" * 90)
    print(f"Overall ExpRate (Exact Match): {matches}/{total} ({accuracy:.1f}%) | Avg Edit Distance: {mean_dist:.2f}\n")

    # Save Markdown Artifact
    report_path = artifact_dir / "results_mth1200_10samples.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BTTR Model Evaluation on 10 MTH1200 Samples\n\n")
        f.write(f"- **Checkpoint**: `{Path(args.ckpt).name}`\n")
        f.write(f"- **Dataset**: `MTH1200` (`{args.data_dir}`)\n")
        f.write(f"- **Exact Match (ExpRate)**: **{matches}/{total} ({accuracy:.1f}%)**\n")
        f.write(f"- **Mean Levenshtein Edit Distance**: **{mean_dist:.2f}**\n\n")

        f.write("## Side-by-Side Comparison\n\n")
        f.write("| # | Crop Image | Ground Truth IDS | Predicted IDS | Status | Edit Dist |\n")
        f.write("|---|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results:
            status_icon = "✅ Match" if r["match"] else "❌ Diff"
            img_rel = r["artifact_img_path"]
            f.write(f"| {r['idx']} | ![{r['filename']}]({img_rel})<br>`{r['filename']}` | **`{r['gt_ids']}`** | **`{r['pred_ids']}`** | {status_icon} | {r['edit_dist']} |\n")

    print(f"Markdown report generated at: {report_path}")

    # Generate PDF Report
    if args.pdf_path:
        from generate_pdf_report import create_pdf
        pdf_out = Path(args.pdf_path)
        if not pdf_out.is_absolute():
            pdf_out = SCRIPT_DIR / pdf_out
        
        try:
            from generate_pdf_report import create_pdf
            create_pdf(
                str(pdf_out),
                results_data=results,
                ckpt_name=Path(args.ckpt).name,
                dataset_name=f"MTH1200",
                images_dir=images_artifact_dir,
            )
            print(f"PDF report generated at: {pdf_out}")
            # Also sync copy to artifact directory if different
            art_pdf = artifact_dir / pdf_out.name
            if art_pdf.resolve() != pdf_out.resolve():
                shutil.copy2(str(pdf_out), str(art_pdf))
                print(f"PDF report copied to artifacts: {art_pdf}")
        except Exception as e:
            print(f"Notice: PDF generation skipped or encountered error: {e}")


if __name__ == "__main__":
    main()