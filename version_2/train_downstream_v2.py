import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.utils import set_seed
from data.datasets import PredictiveMaintenanceDataset
from models.encoder import Encoder
from models.classifier import SimpleClassifier
from utils.evaluation import evaluate


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
set_seed(42)

WINDOW_SIZE = 24  
PATCH_LEN = 6
NUM_PATCHES = WINDOW_SIZE // PATCH_LEN   
LATENT_DIM = 64     
CNN_H_DIM = 64          
NHEAD = 2           
NUM_TRANS_LAYERS = 2

BATCH_SIZE = 256
EPOCHS = 25
FREEZE_EPOCHS = 3

DATA_DIR = "SC-JEPA/data" 
CHECKPOINT_DIR = 'SC-JEPA/checkpoints'

print("[INFO] Preparing data for downstream classification...")
train_csv = os.path.join(DATA_DIR, "train.csv")
val_csv = os.path.join(DATA_DIR, "val.csv")
test_csv = os.path.join(DATA_DIR, "test.csv")

train_ds = PredictiveMaintenanceDataset(train_csv, window_size=WINDOW_SIZE, mode='downstream')
x_sample, _ = train_ds[0]
IN_CHANNELS = x_sample.shape[1]

val_ds = PredictiveMaintenanceDataset(val_csv, window_size=WINDOW_SIZE, mode='downstream', scaler=train_ds.scaler)

labels = train_ds.window_labels()
n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
class_weight = torch.tensor([1.0, n_neg / n_pos], dtype=torch.float32, device=DEVICE)
print(f"[INFO] Train windows {len(train_ds)} (pos {n_pos}) | Class weights {class_weight.tolist()}")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

print("[INFO] Initializing models and loading pre-trained encoder weights...")
encoder = Encoder(
    num_patches=NUM_PATCHES, patch_len=PATCH_LEN, latent_dim=LATENT_DIM, 
    cnn_h_dim=CNN_H_DIM, trans_nhead=NHEAD, trans_num_layers=NUM_TRANS_LAYERS, in_channels=IN_CHANNELS
).to(DEVICE)

best_encoder_path = os.path.join(CHECKPOINT_DIR, 'encoder.pth')
ckpt = torch.load(best_encoder_path, map_location=DEVICE, weights_only=True)

if "ema" in ckpt:
    encoder.load_state_dict(ckpt["ema"])
else:
    encoder.load_state_dict(ckpt)

classifier = SimpleClassifier(input_dim=LATENT_DIM, num_patches=NUM_PATCHES).to(DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weight)

optimizer = optim.AdamW([
    {'params': encoder.parameters(), 'lr': 1e-4},
    {'params': classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

best_f1 = -1.0
best_thresh = 0.5
wait = 0
patience = 8
downstream_model_path = os.path.join(CHECKPOINT_DIR, "downstream.pth")

print("[INFO] Starting classifier training...")

for epoch in range(1, EPOCHS + 1):
    frozen = epoch <= FREEZE_EPOCHS
    for p in encoder.parameters():
        p.requires_grad = not frozen
    
    encoder.train(not frozen)
    classifier.train()
    total_loss = 0
    
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}{' (frozen enc)' if frozen else ''}", leave=False):
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)

        logits = classifier(encoder(x))
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(classifier.parameters()) + list(encoder.parameters()), 1.0)
        optimizer.step()
        
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    
    classifier.eval()
    encoder.eval()
    
    eval_res = evaluate(classifier, encoder, val_loader, device=DEVICE)
    
    if isinstance(eval_res, dict):
        f1 = eval_res["f1"]
        val_thresh = eval_res["threshold"]
    else:
        val_thresh, f1 = eval_res[0], eval_res[1]
    
    scheduler.step(f1)
        
    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val F1: {f1:.4f} | Thr: {val_thresh:.3f} | LR Enc: {optimizer.param_groups[0]['lr']:.1e}")

    if f1 > best_f1:
        best_f1 = f1
        best_thresh = val_thresh
        wait = 0
        
        torch.save({
            "classifier_state_dict": classifier.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "threshold": best_thresh,
            "config": {
                "num_patches": NUM_PATCHES, "patch_len": PATCH_LEN, 
                "latent_dim": LATENT_DIM, "cnn_h_dim": CNN_H_DIM, 
                "nhead": NHEAD, "num_layers": NUM_TRANS_LAYERS,
                "in_channels": IN_CHANNELS, "window_size": WINDOW_SIZE
            }
        }, downstream_model_path) 
        print(f"   --> Saving the best model (Val F1: {best_f1:.4f})")
    else:
        wait += 1

    if wait >= patience:
        print(f"[EARLY STOPPING] No improvement for {patience} epochs.")
        break

print(f"\n[INFO] Downstream Complete! Best validation F1: {best_f1:.4f} @ Threshold: {best_thresh:.4f}")