"""
Shared evaluation protocol for SC-JEPA and the baselines.

- the decision threshold is chosen on the VALIDATION set (max F1) and applied unchanged
  to the test set;
- PR-AUC (average precision) is reported because with ~1% positives it is far more
  informative than ROC-AUC;
- the test F1 at the test-optimal threshold ("oracle") is reported for diagnostics only:
  the gap to the val-threshold F1 measures how badly the threshold transfers.
"""
import json
import os

import numpy as np
import torch
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)


def best_threshold(probs, targets):
    """Threshold maximising F1, computed exactly from the precision-recall curve."""
    p, r, t = precision_recall_curve(targets, probs)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.argmax(f1))
    return float(t[i]), float(f1[i])


def metrics_at(probs, targets, thresh):
    pred = (probs >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, pred, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"threshold": float(thresh), "precision": float(precision), "recall": float(recall),
            "f1": float(f1), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def ranking_metrics(probs, targets):
    return {"roc_auc": float(roc_auc_score(targets, probs)),
            "pr_auc": float(average_precision_score(targets, probs))}


def summarize(val_probs, val_targets, test_probs, test_targets):
    """Full protocol: threshold on val -> metrics on test (+ oracle threshold for diagnostics)."""
    thr, val_f1 = best_threshold(val_probs, val_targets)
    out = {"val": {**metrics_at(val_probs, val_targets, thr), **ranking_metrics(val_probs, val_targets)},
           "test": {**metrics_at(test_probs, test_targets, thr), **ranking_metrics(test_probs, test_targets)}}
    oracle_thr, oracle_f1 = best_threshold(test_probs, test_targets)
    out["test"]["oracle_threshold"] = oracle_thr
    out["test"]["oracle_f1"] = oracle_f1
    return out


def print_report(name, res):
    v, t = res["val"], res["test"]
    print(f"\n--- {name} ---")
    print(f"threshold (from val): {t['threshold']:.4f}   val F1 {v['f1']:.4f}   val PR-AUC {v['pr_auc']:.4f}")
    print(f"TEST  precision {t['precision']*100:6.2f}%  recall {t['recall']*100:6.2f}%  F1 {t['f1']*100:6.2f}%"
          f"  PR-AUC {t['pr_auc']:.4f}  ROC-AUC {t['roc_auc']:.4f}")
    print(f"TEST  TN {t['tn']} | FP {t['fp']} | FN {t['fn']} | TP {t['tp']}")
    print(f"TEST  F1 at test-optimal threshold (oracle, diagnostic only): {t['oracle_f1']*100:.2f}% @ {t['oracle_threshold']:.4f}")


def save_result(name, res, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump({"model": name, **res}, f, indent=2)
    return path


# ----------------------------------------------------------------------------- torch helpers
@torch.no_grad()
def predict_probs(classifier, encoder, dataloader, device="cuda"):
    classifier.eval()
    encoder.eval()
    probs, targets = [], []
    for x, y in dataloader:
        logits = classifier(encoder(x.to(device)))
        probs.append(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def evaluate(classifier, encoder, dataloader, device="cuda"):
    """Validation-time evaluation: best threshold + metrics at that threshold + ranking metrics."""
    probs, targets = predict_probs(classifier, encoder, dataloader, device)
    thr, _ = best_threshold(probs, targets)
    return {**metrics_at(probs, targets, thr), **ranking_metrics(probs, targets)}
