import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.utils import set_seed
from data.datasets import PredictiveMaintenanceDataset
from models.encoder import Encoder
from models.classifier import SimpleClassifier
from utils.evaluation import predict_probs, print_report, save_result, summarize

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

DATA_DIR = "SC-JEPA/data"
downstream_model_path = 'SC-JEPA/checkpoints/downstream.pth'
RESULT_NAME = "scjepa_v2_aligned"

print(f"[INFO] Loading checkpoint from {downstream_model_path}...")
ckpt = torch.load(downstream_model_path, map_location=DEVICE, weights_only=False)
cfg = ckpt["config"]

print("[INFO] Preparing datasets...")
train_ds = PredictiveMaintenanceDataset(
    os.path.join(DATA_DIR, "train.csv"), 
    mode="downstream",
    window_size=cfg["window_size"], 
    forecast_horizon=cfg["forecast_horizon"], 
    patch_len=cfg["patch_len"]
)

loaders = {}
for split in ("val", "test"):
    ds = PredictiveMaintenanceDataset(
        os.path.join(DATA_DIR, f"{split}.csv"), 
        mode="downstream", 
        window_size=cfg["window_size"],
        forecast_horizon=cfg["forecast_horizon"], 
        patch_len=cfg["patch_len"], 
        scaler=train_ds.scaler
    )
    loaders[split] = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=4)

print("[INFO] Initializing models based on saved configurations...")
encoder = Encoder(
    num_patches=cfg["num_patches"], patch_len=cfg["patch_len"], latent_dim=cfg["latent_dim"], 
    cnn_h_dim=cfg["cnn_h_dim"], trans_nhead=cfg["nhead"], trans_num_layers=cfg["num_layers"], 
    in_channels=cfg["in_channels"]
).to(DEVICE)

classifier = SimpleClassifier(input_dim=cfg["latent_dim"], num_patches=cfg["num_patches"]).to(DEVICE)

encoder.load_state_dict(ckpt["encoder_state_dict"])
classifier.load_state_dict(ckpt["classifier_state_dict"])
val_threshold = ckpt["threshold"]

print("\n" + "="*50)
print("Starting evaluation on val and test sets...")
print("="*50)

val_probs, val_y = predict_probs(classifier, encoder, loaders["val"], DEVICE)
test_probs, test_y = predict_probs(classifier, encoder, loaders["test"], DEVICE)

res = summarize(val_probs, val_y, test_probs, test_y)

print(f"[INFO] Checkpoint config loaded. Saved validation threshold: {val_threshold:.4f}\n")

print_report(RESULT_NAME, res)
json_path = save_result(RESULT_NAME, res)

print(f"\n[INFO] Evaluation Complete! Results saved to {json_path}")