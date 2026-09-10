import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from data.utils import set_seed
from data.datasets import PredictiveMaintenanceDataset
from models.encoder import Encoder
from models.classifier import SimpleClassifier
from utils.evaluation import test_evaluation

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

WINDOW_SIZE = 24  
PATCH_LEN = 6
NUM_PATCHES = (WINDOW_SIZE // 2) // PATCH_LEN
LATENT_DIM = 64

CNN_H_DIM = 64          
NHEAD = 2           
NUM_TRANS_LAYERS = 2

BATCH_SIZE = 256
DATA_DIR = "/home/jovyan/workspace/SC-JEPA/data"

class DownstreamWrapper(Dataset):
    def __init__(self, original_ds, num_patches, patch_len, in_channels):
        self.ds = original_ds
        self.num_patches = num_patches
        self.patch_len = patch_len
        self.in_channels = in_channels
        
        y_matrix = self.ds.df[self.ds.label_cols].values
        labels = []
        for seq in self.ds.sequences:
            y_window = y_matrix[seq['label_start'] : seq['label_end']]
            y_label = np.zeros(1) if y_window.size == 0 else np.max(y_window, axis=0)
            y_expanded = torch.tensor(y_label, dtype=torch.float32).repeat(self.num_patches).to(torch.long)
            labels.append(y_expanded)
        self.y_label = torch.stack(labels)

    def __len__(self): 
        return len(self.ds)

    def __getitem__(self, idx):
        x, _ = self.ds[idx]
        y_expanded = self.y_label[idx]
        x_recent = x[-(self.num_patches * self.patch_len):, :]
        return x_recent.view(self.num_patches, self.patch_len, self.in_channels), y_expanded

print("[INFO] Loading test data...")
train_csv = os.path.join(DATA_DIR, "train.csv")
test_csv = os.path.join(DATA_DIR, "test.csv")

train_ds_scaler = PredictiveMaintenanceDataset(train_csv, window_size=WINDOW_SIZE, mode='downstream')

x_sample, _ = train_ds_scaler[0]
IN_CHANNELS = x_sample.shape[1]

test_ds_raw = PredictiveMaintenanceDataset(test_csv, window_size=WINDOW_SIZE, mode='downstream', scaler=train_ds_scaler.scaler)

test_ds_wrapped = DownstreamWrapper(test_ds_raw, NUM_PATCHES, PATCH_LEN, IN_CHANNELS)
test_loader_down = DataLoader(test_ds_wrapped, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print("[INFO] Initializing empty models...")
encoder = Encoder(
    num_patches=NUM_PATCHES, patch_len=PATCH_LEN, latent_dim=LATENT_DIM, 
    cnn_h_dim=CNN_H_DIM, trans_nhead=NHEAD, trans_num_layers=NUM_TRANS_LAYERS, in_channels=IN_CHANNELS
).to(DEVICE)

classifier = SimpleClassifier(input_dim=LATENT_DIM, num_patches=NUM_PATCHES).to(DEVICE)

downstream_model_path = '/home/jovyan/workspace/SC-JEPA/checkpoints/downstream.pth'
print(f"[INFO] Loading weights and optimal threshold from {downstream_model_path}...")

ckpt = torch.load(downstream_model_path, map_location=DEVICE, weights_only=False)

encoder.load_state_dict(ckpt["encoder_state_dict"])
classifier.load_state_dict(ckpt["classifier_state_dict"])
val_threshold = ckpt["threshold"]

encoder.eval()
classifier.eval()

print("\n" + "="*50)
print("Starting evaluation on test set...")
print("="*50)

print(f"[INFO] Using the optimal threshold computed during validation: {val_threshold:.4f}\n")

test_evaluation(classifier, encoder, test_loader_down, val_threshold, device=DEVICE)

print("\n[INFO] Evaluation Complete!")