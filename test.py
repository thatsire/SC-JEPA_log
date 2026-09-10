"""
Test-set evaluation of a downstream checkpoint with the shared protocol
(threshold from validation, PR-AUC, oracle F1). Writes results/<name>.json.
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from data.datasets import PredictiveMaintenanceDataset
from data.utils import set_seed
from models.classifier import SimpleClassifier
from models.encoder import Encoder
from utils.evaluation import predict_probs, print_report, save_result, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="dataset")
ap.add_argument("--ckpt", default="checkpoints/downstream.pth")
ap.add_argument("--name", default="scjepa", help="name of the results/<name>.json file")
args = ap.parse_args()

set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
cfg = ckpt["config"]

train_ds = PredictiveMaintenanceDataset(os.path.join(args.data_dir, "train.csv"), mode="downstream",
                                        window_size=cfg["window_size"], forecast_horizon=cfg["forecast_horizon"], patch_len=cfg["patch_len"])
loaders = {}
for split in ("val", "test"):
    ds = PredictiveMaintenanceDataset(os.path.join(args.data_dir, f"{split}.csv"), mode="downstream", window_size=cfg["window_size"],
                                      forecast_horizon=cfg["forecast_horizon"], patch_len=cfg["patch_len"], scaler=train_ds.scaler)
    loaders[split] = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=4)

encoder = Encoder(num_patches=cfg["num_patches"], patch_len=cfg["patch_len"], latent_dim=cfg["latent_dim"], cnn_h_dim=cfg["cnn_h_dim"],
                  trans_nhead=cfg["nhead"], trans_num_layers=cfg["num_layers"], in_channels=cfg["in_channels"]).to(DEVICE)
classifier = SimpleClassifier(input_dim=cfg["latent_dim"], num_patches=cfg["num_patches"]).to(DEVICE)
encoder.load_state_dict(ckpt["encoder_state_dict"])
classifier.load_state_dict(ckpt["classifier_state_dict"])

val_probs, val_y = predict_probs(classifier, encoder, loaders["val"], DEVICE)
test_probs, test_y = predict_probs(classifier, encoder, loaders["test"], DEVICE)
res = summarize(val_probs, val_y, test_probs, test_y)
print(f"[INFO] checkpoint {args.ckpt} (from_scratch={cfg.get('from_scratch', False)}), saved val threshold {ckpt['threshold']:.4f}")
print_report(args.name, res)
print("saved:", save_result(args.name, res))
