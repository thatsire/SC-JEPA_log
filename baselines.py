"""
Baselines (LSTM, Random Forest, K-Means) trained and evaluated with EXACTLY the same
windows, labels, split and threshold protocol as SC-JEPA:
  input 24h of telemetry (+ age, model), label = any failure in the following 12h,
  threshold chosen on val (max F1), applied on test. Writes results/<model>.json.

RF / K-Means use per-window summary statistics (mean, std, min, max, last value, trend of
each telemetry channel + age + model one-hot). K-Means is unsupervised: each cluster is
scored with its positive rate on the training set and the score is thresholded on val.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from data.datasets import PredictiveMaintenanceDataset
from data.utils import set_seed
from utils.evaluation import best_threshold, print_report, save_result, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="dataset")
ap.add_argument("--models", default="kmeans,random_forest,lstm")
ap.add_argument("--lstm_epochs", type=int, default=20)
ap.add_argument("--n_jobs", type=int, default=32)
args = ap.parse_args()
set_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WINDOW_SIZE, FORECAST_HORIZON, PATCH_LEN = 24, 12, 6
print("[INFO] Loading windows...")
splits, scaler = {}, None
for split in ("train", "val", "test"):
    ds = PredictiveMaintenanceDataset(os.path.join(args.data_dir, f"{split}.csv"), mode="downstream", window_size=WINDOW_SIZE,
                                      forecast_horizon=FORECAST_HORIZON, patch_len=PATCH_LEN, scaler=scaler)
    scaler = ds.scaler
    splits[split] = (ds.windows(), ds.window_labels())   # (n, 24, 9), (n,)
    print(f"  {split}: {splits[split][0].shape}, positives {int(splits[split][1].sum())}")
(W_tr, y_tr), (W_va, y_va), (W_te, y_te) = splits["train"], splits["val"], splits["test"]


def window_features(W):
    tel = W[:, :, :4]                                     # volt, rotate, pressure, vibration
    feats = [tel.mean(1), tel.std(1), tel.min(1), tel.max(1), tel[:, -1, :],
             tel[:, -6:, :].mean(1) - tel[:, :6, :].mean(1)]   # trend: last 6h vs first 6h
    return np.concatenate(feats + [W[:, -1, 4:5], W[:, -1, 5:9]], axis=1).astype(np.float32)  # 29 features


models = args.models.split(",")
if "random_forest" in models or "kmeans" in models:
    F_tr, F_va, F_te = window_features(W_tr), window_features(W_va), window_features(W_te)

if "kmeans" in models:
    print("\n[K-MEANS] fitting...")
    fs = StandardScaler().fit(F_tr)
    km = KMeans(n_clusters=16, n_init=5, random_state=42).fit(fs.transform(F_tr))
    c_tr = km.labels_
    pos_rate = np.bincount(c_tr, weights=y_tr, minlength=16) / np.maximum(np.bincount(c_tr, minlength=16), 1)
    print("  cluster positive rates (train):", np.round(pos_rate, 4))
    s_va, s_te = pos_rate[km.predict(fs.transform(F_va))], pos_rate[km.predict(fs.transform(F_te))]
    res = summarize(s_va, y_va, s_te, y_te)
    print_report("kmeans", res); save_result("kmeans", res)

if "random_forest" in models:
    print("\n[RANDOM FOREST] fitting...")
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                n_jobs=args.n_jobs, random_state=42).fit(F_tr, y_tr)
    res = summarize(rf.predict_proba(F_va)[:, 1], y_va, rf.predict_proba(F_te)[:, 1], y_te)
    print_report("random_forest", res); save_result("random_forest", res)

if "lstm" in models:
    print("\n[LSTM] training...")

    class LSTMClassifier(nn.Module):
        def __init__(self, in_dim, hidden=64):
            super().__init__()
            self.lstm = nn.LSTM(in_dim, hidden, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 2))

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1])

    Xtr, Ytr = torch.from_numpy(W_tr).to(DEVICE), torch.from_numpy(y_tr).to(DEVICE)
    Xva, Xte = torch.from_numpy(W_va).to(DEVICE), torch.from_numpy(W_te).to(DEVICE)
    model = LSTMClassifier(W_tr.shape[2]).to(DEVICE)
    weight = torch.tensor([1.0, (y_tr == 0).sum() / max(y_tr.sum(), 1)], dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    @torch.no_grad()
    def probs(X):
        model.eval()
        return torch.cat([torch.softmax(model(X[i:i + 4096]), 1)[:, 1] for i in range(0, len(X), 4096)]).cpu().numpy()

    best_f1, best_state, wait, BS = -1.0, None, 0, 512
    for epoch in range(1, args.lstm_epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        tot = 0.0
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]
            loss = crit(model(Xtr[idx]), Ytr[idx])
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item()
        thr, f1 = best_threshold(probs(Xva), y_va)
        print(f"  epoch {epoch} loss {tot / (len(perm) // BS + 1):.4f}  val F1 {f1:.4f} @ {thr:.3f}")
        if f1 > best_f1:
            best_f1, wait = f1, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= 5:
                print("  early stopping"); break
    model.load_state_dict(best_state)
    res = summarize(probs(Xva), y_va, probs(Xte), y_te)
    print_report("lstm", res); save_result("lstm", res)
