import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset

from data.utils import set_seed
from data.datasets import PredictiveMaintenanceDataset
from models.encoder import Encoder
from models.classifier import SimpleClassifier, FocalLoss
from utils.evaluation import evaluate


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
set_seed(42)

WINDOW_SIZE = 24  
PATCH_LEN = 6
NUM_PATCHES = (WINDOW_SIZE // 2) // PATCH_LEN   
LATENT_DIM = 64     
CNN_H_DIM = 64          
NHEAD = 2           
NUM_TRANS_LAYERS = 2

BATCH_SIZE = 256
EPOCHS = 50

DATA_DIR = "/home/jovyan/workspace/SC-JEPA/data" 
CHECKPOINT_DIR = '/home/jovyan/workspace/SC-JEPA/checkpoints'

class DownstreamWrapper(Dataset):
    def __init__(self, original_ds, num_patches, patch_len, in_channels):
        self.ds = original_ds
        self.num_patches = num_patches
        self.patch_len = patch_len
        self.in_channels = in_channels
        
        print(f"[INFO] Extracting labels for {len(self.ds)} sequences...")
        y_matrix = self.ds.df[self.ds.label_cols].values
        labels = []
        
        for seq in self.ds.sequences:
            y_window = y_matrix[seq['label_start'] : seq['label_end']]
            y_label = np.zeros(1) if y_window.size == 0 else np.max(y_window, axis=0)
            y_tensor = torch.tensor(y_label, dtype=torch.float32)
            y_expanded = y_tensor.repeat(self.num_patches).to(torch.long)
            labels.append(y_expanded)
            
        self.y_label = torch.stack(labels)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        y_expanded = self.y_label[idx]
        past_hours = self.num_patches * self.patch_len
        x_recent = x[-past_hours:, :]
        x_reshaped = x_recent.view(self.num_patches, self.patch_len, self.in_channels)
        return x_reshaped, y_expanded

print("[INFO] Preparing data for downstream classification...")
train_csv = os.path.join(DATA_DIR, "train.csv")
val_csv = os.path.join(DATA_DIR, "val.csv")
test_csv = os.path.join(DATA_DIR, "test.csv")

train_ds_downstream = PredictiveMaintenanceDataset(train_csv, window_size=WINDOW_SIZE, mode='downstream')
x_sample, _ = train_ds_downstream[0]
IN_CHANNELS = x_sample.shape[1]
val_ds_downstream = PredictiveMaintenanceDataset(val_csv, window_size=WINDOW_SIZE, mode='downstream', scaler=train_ds_downstream.scaler)
test_ds_downstream = PredictiveMaintenanceDataset(test_csv, window_size=WINDOW_SIZE, mode='downstream', scaler=train_ds_downstream.scaler)

print("[INFO] Applying Downstream Wrapper...")
train_ds_wrapped = DownstreamWrapper(train_ds_downstream, NUM_PATCHES, PATCH_LEN, IN_CHANNELS)
val_ds_wrapped = DownstreamWrapper(val_ds_downstream, NUM_PATCHES, PATCH_LEN, IN_CHANNELS)
test_ds_wrapped = DownstreamWrapper(test_ds_downstream, NUM_PATCHES, PATCH_LEN, IN_CHANNELS)

print(f"Downstream data ready! Train: {len(train_ds_wrapped)}, Val: {len(val_ds_wrapped)}, Test: {len(test_ds_wrapped)}")

train_loader_down = DataLoader(train_ds_wrapped, batch_size=BATCH_SIZE, shuffle = True, num_workers=4, pin_memory=True)
val_loader_down = DataLoader(val_ds_wrapped, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
test_loader_down = DataLoader(test_ds_wrapped, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

print("[INFO] Initializing models and loading pre-trained encoder weights...")
encoder = Encoder(
    num_patches=NUM_PATCHES, patch_len=PATCH_LEN, latent_dim=LATENT_DIM, 
    cnn_h_dim=CNN_H_DIM, trans_nhead=NHEAD, trans_num_layers=NUM_TRANS_LAYERS, in_channels=IN_CHANNELS
).to(DEVICE)

best_encoder_path = os.path.join(CHECKPOINT_DIR, 'encoder.pth')
encoder.load_state_dict(torch.load(best_encoder_path, map_location=DEVICE, weights_only=True))

classifier = SimpleClassifier(input_dim=LATENT_DIM, num_patches=NUM_PATCHES).to(DEVICE)
criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW([
    {'params': encoder.parameters(), 'lr': 5e-5},
    {'params': classifier.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

wandb.init(
    project="SC-JEPA",
    entity="irene-barbagallo03-universit-degli-studi-di-catania",
    name="scjepa-downstream",
    config={
        "epochs": EPOCHS,
        "learning_rate_encoder": 5e-5,
        "learning_rate_classifier": 1e-3,
        "batch_size": BATCH_SIZE,
        "window_size": WINDOW_SIZE
    }
)
best_val_loss = float('inf')
best_thresh = 0.5
wait = 0
patience = 10
downstream_model_path = os.path.join(CHECKPOINT_DIR, "downstream.pth")

print("[INFO] Starting classifier training...")

for epoch in range(1, EPOCHS + 1):
    classifier.train()
    encoder.train()
    total_loss = 0
    
    for x_patch, y_label in tqdm(train_loader_down, desc=f"Epoch {epoch}/{EPOCHS}", leave=False):
        x_patch, y_label = x_patch.to(DEVICE), y_label.to(DEVICE)

        feats = encoder(x_patch)
        logits = classifier(feats)
        
        y_win = y_label.max(dim=1).values
        loss = criterion(logits, y_win)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(classifier.parameters()) + list(encoder.parameters()), 1.0)
        optimizer.step()
        
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader_down)
    
    classifier.eval()
    encoder.eval()
    total_val_loss = 0.0
    
    with torch.no_grad():
        for x_val, y_val in val_loader_down:
            x_val, y_val = x_val.to(DEVICE), y_val.to(DEVICE)
            feats = encoder(x_val)
            logits = classifier(feats)
            y_win = y_val.max(dim=1).values
            val_batch_loss = criterion(logits, y_win)
            total_val_loss += val_batch_loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader_down)
    
    val_thresh, f1, _, val_auc, _ = evaluate(classifier, encoder, val_loader_down, return_metrics=True, device=DEVICE)
    
    # Step scheduler based on Validation Loss
    scheduler.step(avg_val_loss)
    
    wandb.log({
        "downstream/epoch": epoch,
        "downstream/train_loss": avg_train_loss,
        "downstream/val_loss": avg_val_loss,
        "downstream/val_f1": f1,
        "downstream/val_auc": val_auc
    })
    
    print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val F1: {f1:.4f} | Val AUC: {val_auc:.4f}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_thresh = val_thresh
        wait = 0
        
        torch.save({
            "classifier_state_dict": classifier.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "threshold": best_thresh,
        }, downstream_model_path) 
        print(f"   --> Saving the best model (Val Loss: {best_val_loss:.4f})")
    else:
        wait += 1

    if wait >= patience:
        print(f"[EARLY STOPPING] No improvement for {patience} epochs.")
        break

wandb.finish()
print(f"\n[INFO] Downstream Complete! Best validation threshold: {best_thresh:.4f}")