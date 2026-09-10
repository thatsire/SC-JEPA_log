"""Print the comparison table from results/*.json (and write results/comparison.md)."""
import glob
import json
import os

ORDER = ["kmeans", "random_forest", "lstm", "scjepa_scratch", "scjepa", "scjepa_v2"]
NAMES = {"kmeans": "K-Means", "random_forest": "Random Forest", "lstm": "LSTM",
         "scjepa_scratch": "SC-JEPA encoder from scratch (ablation)",
         "scjepa": "SC-JEPA pre-trained, run 1 (early-stopped before codebook collapse)",
         "scjepa_v2": "SC-JEPA pre-trained, run 2 (dead-code reset, stable codebook)"}
rows = {}
for path in glob.glob("results/*.json"):
    r = json.load(open(path))
    rows[r["model"]] = r
names = [m for m in ORDER if m in rows] + sorted(set(rows) - set(ORDER))
lines = ["| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | F1 @ test-optimal thr |",
         "|---|---|---|---|---|---|---|"]
for m in names:
    t = rows[m]["test"]
    lines.append(f"| {NAMES.get(m, m)} | {t['precision']*100:.2f}% | {t['recall']*100:.2f}% | {t['f1']*100:.2f}% | "
                 f"{t['pr_auc']:.3f} | {t['roc_auc']:.3f} | {t['oracle_f1']*100:.2f}% |")
table = "\n".join(lines)
print(table)
os.makedirs("results", exist_ok=True)
open("results/comparison.md", "w").write("Test set, threshold chosen on validation (max F1).\n\n" + table + "\n")
