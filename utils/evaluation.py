import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
from tqdm import tqdm

@torch.no_grad()
def evaluate(classifier, encoder, dataloader, return_metrics=False, device='cuda'):

    classifier.eval()
    encoder.eval()
    
    all_preds = []
    all_targets = []
    
    for x, y in dataloader:
        x = x.to(device)
        
        feats = encoder(x)
        logits = classifier(feats)
        
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        all_preds.extend(probs.cpu().numpy())
        
        y_win = y.max(dim=1).values
        all_targets.extend(y_win.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    auc = roc_auc_score(all_targets, all_preds)
    
    # Best threshold search
    best_f1 = 0
    best_thresh = 0.5
    min_prob = np.min(all_preds)
    max_prob = np.max(all_preds)
    
    thresholds = np.linspace(max(0.001, min_prob), min(0.999, max_prob), 100)
    
    for thresh in thresholds:
        preds_bin = (all_preds >= thresh).astype(int)
        f1 = f1_score(all_targets, preds_bin, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    if return_metrics:
        preds_bin = (all_preds >= best_thresh).astype(int)
        cm = confusion_matrix(all_targets, preds_bin)
        return best_thresh, best_f1, cm, auc, (all_preds, all_targets)
    
    return best_thresh, best_f1

@torch.no_grad()
def test_evaluation(classifier, encoder, dataloader, threshold, device='cuda'):

    classifier.eval()
    encoder.eval()
    
    all_preds = []
    all_targets = []
    
    for x, y in tqdm(dataloader, desc="Test Set Evaluation"):
        x = x.to(device)
        
        feats = encoder(x)
        logits = classifier(feats)
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        all_preds.extend(probs.cpu().numpy())
        
        y_win = y.max(dim=1).values
        all_targets.extend(y_win.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    auc = roc_auc_score(all_targets, all_preds)
    preds_bin = (all_preds >= threshold).astype(int)
    
    print("\n--- Test set results ---")
    print(f"ROC AUC Score: {auc:.4f}")
    print(f"Threshold applied: {threshold:.4f}\n")
    
    print("Classification Report:")
    print(classification_report(all_targets, preds_bin, digits=4))
    
    cm = confusion_matrix(all_targets, preds_bin)
    print("\nConfusion Matrix:")
    print(f"True Negative (TN): {cm[0][0]} | False Positive (FP): {cm[0][1]}")
    print(f"False Negative (FN): {cm[1][0]} | True Positive (TP): {cm[1][1]}")