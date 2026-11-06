import os
import csv
import json
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Setup styling for presentation-quality figures
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 2.2
plt.rcParams['lines.markersize'] = 6

OUTPUT_DIR = "/home/projectg7/workspace/Week6_Recognition/Project Dr.Cuong/BTTR-2.0/training_report"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = "/home/projectg7/workspace/Week6_Recognition/Project Dr.Cuong/BTTR-2.0/lightning_logs/version_7/metrics.csv"

with open(CSV_PATH, 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

epochs_data = defaultdict(dict)
for r in rows:
    ep = r.get('epoch')
    if ep != '':
        ep = int(float(ep))
        for k, v in r.items():
            if v != '' and k != 'epoch':
                epochs_data[ep][k] = float(v)

# Also capture lr changes
lr_by_epoch = {}
current_lr = 0.1
for ep in range(20):
    if ep >= 16:
        current_lr = 0.001
    elif ep >= 8:
        current_lr = 0.01
    else:
        current_lr = 0.1
    lr_by_epoch[ep] = current_lr

# Convert to 1-based epoch for presentation (Epoch 1 to 20)
epochs = [ep + 1 for ep in sorted(epochs_data.keys())]
train_losses = [epochs_data[ep]['train_loss'] for ep in sorted(epochs_data.keys())]
val_losses = [epochs_data[ep]['val_loss'] for ep in sorted(epochs_data.keys())]
val_exprates = [epochs_data[ep]['val_ExpRate'] * 100.0 for ep in sorted(epochs_data.keys())]
lrs = [lr_by_epoch[ep] for ep in sorted(epochs_data.keys())]

# Colors
C_TRAIN = "#2563eb"   # Royal Blue
C_VAL = "#dc2626"     # Crimson Red
C_ACC = "#059669"     # Emerald Green
C_LR = "#d97706"      # Amber Orange
C_BG_ACCENT = "#f3f4f6"

# -------------------------------------------------------------
# 1. Loss Convergence Plot
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
ax.plot(epochs, train_losses, marker='o', color=C_TRAIN, label='Training Loss', lw=2.4)
ax.plot(epochs, val_losses, marker='s', color=C_VAL, label='Validation Loss', lw=2.4)

# Shade LR regions
ax.axvspan(1, 8.5, color='#3b82f6', alpha=0.07, label='LR = 0.10 (Phase 1)')
ax.axvspan(8.5, 16.5, color='#10b981', alpha=0.07, label='LR = 0.01 (Phase 2)')
ax.axvspan(16.5, 20, color='#8b5cf6', alpha=0.07, label='LR = 0.001 (Phase 3)')

# Highlight best validation loss
min_val_loss = min(val_losses)
min_val_ep = epochs[val_losses.index(min_val_loss)]
ax.annotate(f'Best Val Loss: {min_val_loss:.4f}\n(Epoch {min_val_ep})',
            xy=(min_val_ep, min_val_loss),
            xytext=(min_val_ep - 3.5, min_val_loss + 0.05),
            arrowprops=dict(facecolor=C_VAL, shrink=0.05, width=1.5, headwidth=7),
            fontweight='bold', color=C_VAL,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=C_VAL, lw=1.2))

ax.set_title("BTTR Training & Validation Loss across Epochs", pad=12, fontweight='bold')
ax.set_xlabel("Epoch (1 to 20)", labelpad=8)
ax.set_ylabel("Cross-Entropy Loss", labelpad=8)
ax.set_xticks(epochs)
ax.set_xlim(0.5, 20.5)
ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='upper right')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.png"), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, "loss_curves.pdf"))
plt.close()

# -------------------------------------------------------------
# 2. Validation ExpRate (Accuracy) Plot
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5), dpi=300)
ax.plot(epochs, val_exprates, marker='^', color=C_ACC, label='Validation ExpRate (%)', lw=2.5)

# Highlight best ExpRate
max_acc = max(val_exprates)
max_acc_ep = epochs[val_exprates.index(max_acc)]
ax.scatter([max_acc_ep], [max_acc], color='#b91c1c', s=120, zorder=5, label=f'Peak: {max_acc:.2f}% (Epoch {max_acc_ep})')

ax.annotate(f'Peak Accuracy: {max_acc:.2f}%\n(Epoch {max_acc_ep}, Step 504k)',
            xy=(max_acc_ep, max_acc),
            xytext=(max_acc_ep - 5, max_acc - 1.2),
            arrowprops=dict(facecolor='#b91c1c', shrink=0.05, width=1.5, headwidth=7),
            fontweight='bold', color='#991b1b',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec='#b91c1c', lw=1.2))

# Reference baseline
ax.axhline(88.30, color='#9ca3af', linestyle=':', lw=1.5, label='Initial Epoch ExpRate (88.30%)')

ax.set_title("Nom Character Recognition Accuracy (Validation ExpRate)", pad=12, fontweight='bold')
ax.set_xlabel("Epoch (1 to 20)", labelpad=8)
ax.set_ylabel("ExpRate (%)", labelpad=8)
ax.set_xticks(epochs)
ax.set_xlim(0.5, 20.5)
ax.set_ylim(87.5, 92.0)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=1))
ax.legend(frameon=True, facecolor='white', framealpha=0.95, loc='lower right')
ax.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "exprate_accuracy.png"), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, "exprate_accuracy.pdf"))
plt.close()

# -------------------------------------------------------------
# 3. Learning Rate Progression Plot
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 4), dpi=300)
ax.step(epochs, lrs, where='mid', color=C_LR, lw=2.5, label='Adadelta Learning Rate')
ax.scatter(epochs, lrs, color=C_LR, s=40, zorder=4)

ax.set_yscale('log')
ax.set_title("Adaptive Learning Rate Schedule (ReduceLROnPlateau)", pad=12, fontweight='bold')
ax.set_xlabel("Epoch (1 to 20)", labelpad=8)
ax.set_ylabel("Learning Rate (log scale)", labelpad=8)
ax.set_xticks(epochs)
ax.set_xlim(0.5, 20.5)
ax.grid(True, which="both", ls="--", alpha=0.5)

ax.text(4.5, 0.12, "Stage 1: lr = 0.10\n(Warmup & Global Optimization)", color='#92400e', fontweight='bold', ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="#fef3c7", ec="#f59e0b", lw=1))
ax.text(12, 0.012, "Stage 2: lr = 0.01\n(Fine-tuning / Plateau Trigger)", color='#065f46', fontweight='bold', ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="#d1fae5", ec="#10b981", lw=1))
ax.text(18.5, 0.0012, "Stage 3: lr = 0.001\n(Convergence)", color='#4c1d95', fontweight='bold', ha='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="#ede9fe", ec="#8b5cf6", lw=1))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "learning_rate.png"), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, "learning_rate.pdf"))
plt.close()

# -------------------------------------------------------------
# 4. Multi-Panel Comprehensive Dashboard (3-in-1 for Audience)
# -------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True, dpi=300)

# Panel 1: Loss curves
ax1.plot(epochs, train_losses, marker='o', color=C_TRAIN, label='Train Loss', lw=2.2)
ax1.plot(epochs, val_losses, marker='s', color=C_VAL, label='Val Loss', lw=2.2)
ax1.set_ylabel("Loss (Cross-Entropy)", fontweight='bold')
ax1.set_title("(A) Training & Validation Loss Evolution", loc='left', fontweight='bold', color='#1e3a8a')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend(loc='upper right', frameon=True, facecolor='white')
ax1.axvspan(1, 8.5, color='#3b82f6', alpha=0.06)
ax1.axvspan(8.5, 16.5, color='#10b981', alpha=0.06)
ax1.axvspan(16.5, 20, color='#8b5cf6', alpha=0.06)

# Panel 2: Accuracy
ax2.plot(epochs, val_exprates, marker='^', color=C_ACC, label='Val ExpRate (%)', lw=2.2)
ax2.scatter([max_acc_ep], [max_acc], color='#b91c1c', s=100, zorder=5)
ax2.annotate(f'Peak: {max_acc:.2f}% (Epoch {max_acc_ep})',
             xy=(max_acc_ep, max_acc), xytext=(max_acc_ep - 4.5, max_acc - 1.1),
             arrowprops=dict(facecolor='#b91c1c', shrink=0.05, width=1.2, headwidth=6),
             fontweight='bold', color='#991b1b',
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec='#b91c1c', lw=1))
ax2.set_ylabel("ExpRate (%)", fontweight='bold')
ax2.set_title("(B) Model Recognition Accuracy (ExpRate)", loc='left', fontweight='bold', color='#1e3a8a')
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100, decimals=1))
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(loc='lower right', frameon=True, facecolor='white')
ax2.axvspan(1, 8.5, color='#3b82f6', alpha=0.06)
ax2.axvspan(8.5, 16.5, color='#10b981', alpha=0.06)
ax2.axvspan(16.5, 20, color='#8b5cf6', alpha=0.06)

# Panel 3: Learning Rate
ax3.step(epochs, lrs, where='mid', color=C_LR, lw=2.2, label='Learning Rate')
ax3.scatter(epochs, lrs, color=C_LR, s=35, zorder=4)
ax3.set_yscale('log')
ax3.set_ylabel("Learning Rate", fontweight='bold')
ax3.set_title("(C) Dynamic Learning Rate Schedule", loc='left', fontweight='bold', color='#1e3a8a')
ax3.set_xlabel("Training Epoch (1 to 20)", fontweight='bold', labelpad=8)
ax3.set_xticks(epochs)
ax3.set_xlim(0.5, 20.5)
ax3.grid(True, which="both", ls="--", alpha=0.5)
ax3.legend(loc='lower left', frameon=True, facecolor='white')

plt.suptitle("BTTR Model Training Performance & Metric Progression (Epoch 1 to 20)",
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "dashboard_metrics.png"), dpi=300)
plt.savefig(os.path.join(OUTPUT_DIR, "dashboard_metrics.pdf"))
plt.close()

# Export a clean JSON summary for LaTeX table generation
summary_data = []
for ep in sorted(epochs_data.keys()):
    summary_data.append({
        "epoch": ep + 1,
        "step": int(epochs_data[ep]['step']),
        "train_loss": round(epochs_data[ep]['train_loss'], 4),
        "val_loss": round(epochs_data[ep]['val_loss'], 4),
        "val_exprate": round(epochs_data[ep]['val_ExpRate'] * 100.0, 2),
        "learning_rate": lr_by_epoch[ep]
    })

with open(os.path.join(OUTPUT_DIR, "metrics_summary.json"), "w") as f:
    json.dump(summary_data, f, indent=2)

print("Generated all plots and metrics_summary.json successfully.")

