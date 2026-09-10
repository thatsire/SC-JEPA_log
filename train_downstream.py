"""
Downstream failure prediction: encoder (pre-trained SC-JEPA or from scratch) + classifier.

Changes with respect to the original version:
  * input pipeline identical to pre-training (same Dataset, same scaler, no instance norm);
  * 24h of input (4 patches) instead of the last 12h of a 24h window;
  * class-weighted cross-entropy (positives are ~1% of the windows);
  * LR scheduler and model selection driven by validation F1 (the original stepped a
    mode='max' scheduler with the validation LOSS, halving the LR every 3 epochs, and
    selected the checkpoint by CE loss, which favours a majority-class predictor);
  * optional warm-up with a frozen encoder before full fine-tuning;
  * --from_scratch ablation to measure what pre-training actually contributes.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import PredictiveMaintenanceDataset
from data.utils import set_seed
from models.classifier import SimpleClassifier
from models.encoder import Encoder
from utils.evaluation import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="dataset")
ap.add_argument("--ckpt_dir", default="checkpoints")
ap.add_argument("--encoder_ckpt", default=None, help="defaults to <ckpt_dir>/encoder.pth")
ap.add_argument("--encoder_key", default="ema", choices=["ema", "online"])
ap.add_argument("--from_scratch", action="store_true", help="ablation: random init, no pre-training")
ap.add_argument("--freeze_epochs", type=int, default=3, help="epochs with frozen encoder before fine-tuning")
ap.add_argument("--epochs", type=int, default=25)
ap.add_argument("--patience", type=int, default=8)
ap.add_argument("--batch_size", type=int, default=256)
ap.add_argument("--lr_encoder", type=float, default=1e-4)
ap.add_argument("--lr_classifier", type=float, default=1e-3)
ap.add_argument("--out_name", default="downstream", help="checkpoint name inside ckpt_dir")
ap.add_argument("--wandb", action="store_true")
args = ap.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
set_seed(42)

WINDOW_SIZE = 24           # hours of input
FORECAST_HORIZON = 12      # failure within the next 12h
PATCH_LEN = 6
NUM_PATCHES = WINDOW_SIZE // PATCH_LEN   # 4, same as pre-training (24h past)
LATENT_DIM, CNN_H_DIM, NHEAD, NUM_TRANS_LAYERS = 64, 64, 2, 2

print("[INFO] Preparing data...")
train_ds = PredictiveMaintenanceDataset(os.path.join(args.data_dir, "train.csv"), mode="downstream",
                                        window_size=WINDOW_SIZE, forecast_horizon=FORECAST_HORIZON, patch_len=PATCH_LEN)
val_ds = PredictiveMaintenanceDataset(os.path.join(args.data_dir, "val.csv"), mode="downstream", window_size=WINDOW_SIZE,
                                      forecast_horizon=FORECAST_HORIZON, patch_len=PATCH_LEN, scaler=train_ds.scaler)
IN_CHANNELS = train_ds.in_channels
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)

labels = train_ds.window_labels()
n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
class_weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=DEVICE)
print(f"[INFO] Train windows {len(train_ds)} (pos {n_pos} = {100*n_pos/len(labels):.2f}%)  Val windows {len(val_ds)}  "
      f"class weights {class_weight.tolist()}")

encoder = Encoder(num_patches=NUM_PATCHES, patch_len=PATCH_LEN, latent_dim=LATENT_DIM, cnn_h_dim=CNN_H_DIM,
                  trans_nhead=NHEAD, trans_num_layers=NUM_TRANS_LAYERS, in_channels=IN_CHANNELS).to(DEVICE)
if args.from_scratch:
    print("[INFO] Encoder randomly initialised (ablation, no pre-training)")
    args.freeze_epochs = 0
else:
    path = args.encoder_ckpt or os.path.join(args.ckpt_dir, "encoder.pth")
    ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
    cfg = ckpt["config"]
    assert cfg["in_channels"] == IN_CHANNELS and cfg["num_patches"] == NUM_PATCHES, f"encoder config mismatch: {cfg}"
    encoder.load_state_dict(ckpt[args.encoder_key])
    print(f"[INFO] Loaded pre-trained encoder ({args.encoder_key}) from {path}")

classifier = SimpleClassifier(input_dim=LATENT_DIM, num_patches=NUM_PATCHES).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weight)
optimizer = optim.AdamW([{"params": encoder.parameters(), "lr": args.lr_encoder},
                         {"params": classifier.parameters(), "lr": args.lr_classifier}], weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

if args.wandb:
    import wandb
    wandb.init(project="SC-JEPA", name=f"scjepa-{args.out_name}", config=vars(args))

os.makedirs(args.ckpt_dir, exist_ok=True)
ckpt_path = os.path.join(args.ckpt_dir, f"{args.out_name}.pth")
best_f1, wait = -1.0, 0

print("[INFO] Starting classifier training...")
for epoch in range(1, args.epochs + 1):
    frozen = epoch <= args.freeze_epochs
    for p in encoder.parameters():
        p.requires_grad = not frozen
    encoder.train(not frozen)   # frozen encoder in eval mode -> deterministic features
    classifier.train()

    total_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}{' (frozen enc)' if frozen else ''}", leave=False):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        loss = criterion(classifier(encoder(x)), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(classifier.parameters()) + list(encoder.parameters()), 1.0)
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    val = evaluate(classifier, encoder, val_loader, device=DEVICE)
    scheduler.step(val["f1"])
    print(f"Epoch {epoch} | train loss {avg_train_loss:.4f} | val F1 {val['f1']:.4f} P {val['precision']:.4f} "
          f"R {val['recall']:.4f} PR-AUC {val['pr_auc']:.4f} ROC-AUC {val['roc_auc']:.4f} thr {val['threshold']:.3f} "
          f"| lr enc {optimizer.param_groups[0]['lr']:.1e}")
    if args.wandb:
        wandb.log({"downstream/epoch": epoch, "downstream/train_loss": avg_train_loss,
                   **{f"downstream/val_{k}": v for k, v in val.items()}})

    if val["f1"] > best_f1:
        best_f1, wait = val["f1"], 0
        torch.save({"classifier_state_dict": classifier.state_dict(), "encoder_state_dict": encoder.state_dict(),
                    "threshold": val["threshold"], "val_metrics": val,
                    "config": {"num_patches": NUM_PATCHES, "patch_len": PATCH_LEN, "latent_dim": LATENT_DIM,
                               "cnn_h_dim": CNN_H_DIM, "nhead": NHEAD, "num_layers": NUM_TRANS_LAYERS,
                               "in_channels": IN_CHANNELS, "window_size": WINDOW_SIZE,
                               "forecast_horizon": FORECAST_HORIZON, "from_scratch": args.from_scratch}}, ckpt_path)
        print(f"   --> best val F1 so far, saved to {ckpt_path}")
    else:
        wait += 1
        if wait >= args.patience:
            print(f"[EARLY STOPPING] no val-F1 improvement for {args.patience} epochs")
            break

if args.wandb:
    wandb.finish()
print(f"\n[INFO] Downstream complete. Best val F1 {best_f1:.4f}")
