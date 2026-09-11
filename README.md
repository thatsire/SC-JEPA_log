# SC-JEPA

## Data

* Dataset: Microsoft Azure Predictive Maintenance (100 machines, hourly telemetry for 2015, 761 failures).
* The raw files (`PdM_telemetry.csv`, `PdM_machines.csv`, `PdM_failures.csv`) are not in the working
  tree. `prepare_data.py` recovers them automatically from git history (commit `c728603`) into
  `raw/` when they are missing; alternatively download the dataset and put the three files in `raw/`.
* `prepare_data.py` builds `dataset/{train,val,test}.csv` from them with a temporal split per machine
  (Jan–Aug / Sep–Oct / Nov–Dec 2015); columns: `datetime, machineID, volt, rotate, pressure, vibration,
  age, model (0..3), fail_comp1..fail_comp4` (1 at the failure hour).

## Pipeline (commands actually used for the results below)

```bash
# 1. train/val/test CSVs, temporal split per machine: Jan-Aug / Sep-Oct / Nov-Dec 2015
python prepare_data.py                                    # -> dataset/{train,val,test}.csv (raw files fetched if missing)

# 2. SC-JEPA pre-training (24h past -> 24h future, 4 patches of 6h, ~1 min/epoch on an RTX A6000)
python train_encoder.py --epochs 30                       # -> checkpoints/encoder.pth (online + EMA encoder)

# 3. downstream classifier: failure within the next 12h from 24h of telemetry
python train_downstream.py                                # pre-trained EMA encoder -> checkpoints/downstream.pth
python train_downstream.py --from_scratch --out_name downstream_scratch   # ablation: random-init encoder

# 4. test-set evaluation (threshold chosen on validation), baselines, comparison table
python test.py --ckpt checkpoints/downstream.pth --name scjepa
python test.py --ckpt checkpoints/downstream_scratch.pth --name scjepa_scratch
python baselines.py                                       # K-Means, Random Forest, LSTM on the same windows (~10 min)
python compare.py                                         # prints the table, writes results/comparison.md
```

Useful flags: `--data_dir`, `--ckpt_dir` on every script; `--wandb` to log to Weights & Biases;
`train_downstream.py --encoder_ckpt <path> --encoder_key ema|online --freeze_epochs N --out_name <name>`;
`train_encoder.py --quant_temp T --stride S --patience P`.
Generated data, checkpoints, logs and results are git-ignored (`dataset/`, `checkpoints*/`, `logs/`, `results/`).

The "run 2" model in the table was trained in a second checkpoint directory so that both runs could be
kept: `train_encoder.py --ckpt_dir checkpoints_v2`, then
`train_downstream.py --encoder_ckpt checkpoints_v2/encoder.pth --out_name downstream_v2` and
`test.py --ckpt checkpoints/downstream_v2.pth --name scjepa_v2`.

## Data protocol (shared by every model)

* input: 24 consecutive hours of `volt, rotate, pressure, vibration, age` standardised with the
  training scaler, plus one-hot `model` -> 9 channels;
* label: 1 if any component fails in the 12 hours following the window (no overlap with the input);
* windows never cross machines or split boundaries;
* decision threshold = argmax F1 on validation, applied unchanged to the test set; PR-AUC reported
  because positives are ~1% of the windows.

| Split | Period (2015) | Hours | Failure hours | Windows | Positive windows |
|---|---|---|---|---|---|
| train | 1 Jan – 31 Aug | 582 600 | 493 | 579 100 | 5 679 (0.98%) |
| val | 1 Sep – 31 Oct | 146 400 | 112 | 142 900 | 1 307 (0.91%) |
| test | 1 Nov – 1 Jan | 147 100 | 114 | 143 600 | 1 317 (0.92%) |

Baselines: Random Forest (300 trees, balanced class weights) and K-Means (16 clusters, each scored by
its training positive rate) use 29 per-window statistics (mean, std, min, max, last value, 6h trend of
the four sensors + age + model); the LSTM (2 layers x 64) reads the raw 24 steps with class-weighted CE.

## Results (test set, 10 Sep 2026)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | F1 @ test-optimal thr |
|---|---|---|---|---|---|---|
| K-Means | 8.37% | 54.44% | 14.52% | 0.080 | 0.944 | 14.52% |
| Random Forest | 12.80% | 65.00% | 21.39% | 0.130 | 0.965 | 21.72% |
| LSTM | 14.64% | 66.74% | 24.01% | 0.147 | 0.971 | 24.21% |
| SC-JEPA, encoder from scratch (ablation) | 15.18% | 50.11% | 23.30% | 0.149 | 0.969 | 23.80% |
| SC-JEPA pre-trained, run 1 | 16.04% | 40.77% | 23.02% | 0.159 | 0.967 | 23.47% |
| **SC-JEPA pre-trained, run 2 (current code defaults)** | 15.27% | 55.43% | 23.94% | 0.156 | 0.970 | 24.27% |

Per-model JSON (validation metrics, thresholds, confusion matrices) is in `results/`.

**Run 1 vs run 2.** Same architecture and data; run 1 used sample-entropy weight 0.1 and no dead-code
reset, and its codebook collapsed at epoch 4 (perplexity 62 -> 5 of 64 codes); early stopping kept
the epoch-3 checkpoint. Run 2 (sample-entropy weight 0.01 + dead-code reset every 500 steps, now the
defaults in `train_encoder.py`) kept perplexity at 63.5 for all epochs. Both were stopped at epoch 7.

**What the pre-training contributes.** Little. The from-scratch encoder reaches F1 23.3% vs 23.0% /
23.9% pre-trained. With the encoder frozen (first 3 downstream epochs) the pre-trained features give
val PR-AUC 0.011 (run 1, i.e. random: base rate 0.009) and 0.081 (run 2); fine-tuning does the work.
The pre-training objective (predict the next-24h code distribution) captures the machine's typical
state, not the rare anomaly that precedes a failure.

**Thresholds transfer.** For every model the gap between F1 at the validation threshold and F1 at the
test-optimal threshold is below 0.5 points.

**The thesis table (Table 5.1) is not reproducible** with this leakage-free temporal split: LSTM 87.8%
there vs 24.0% here, Random Forest 31.7% vs 21.4%, K-Means 26.0% vs 14.5%, SC-JEPA 18.7% vs 23.9%.
The original train/val/test CSVs and baseline code were never in the repository; the most likely cause
is a random split of 1h-stride overlapping windows (train and test windows sharing 23 of 24 hours).

## What was fixed with respect to the first version

| Problem | Fix |
|---|---|
| Quantizer temperature 1.0 on cosine distances -> near-uniform code assignment (max prob <= 0.105), trivial JEPA targets, commitment loss pulling the encoder to a constant | temperature 0.1; codebook/commitment losses in the normalised space; batch-entropy weight 1.0, sample-entropy weight 0.01; dead-code reset every 500 steps; perplexity / max-prob / h-std monitors logged every epoch |
| Per-window instance normalisation only in pre-training (`age` always 0), none downstream | one normalisation, in the Dataset, shared by both stages; `age` scaled, `model` one-hot |
| Instance normalisation removed the absolute level, i.e. the failure signal | removed |
| 2 patches (12h) of context, first half of the window discarded downstream | 4 patches, 24h in both stages, same as the baselines |
| Downstream: unweighted CE on 1% positives, checkpoint chosen by CE loss, `ReduceLROnPlateau(mode='max')` fed with a loss (LR halved every 3 epochs) | class-weighted CE, selection and LR schedule on validation F1, frozen-encoder warm-up, `--from_scratch` ablation |
| Threshold search on 100 linspace points, no PR-AUC | exact search on the PR curve, PR-AUC, oracle-threshold F1 as a diagnostic |
| Only the online encoder saved, deprecated `clip_grad_norm`, dead code, unshuffled validation loader | EMA + online saved, `clip_grad_norm_`, cleaned, validation loader shuffled with a fixed seed |
| No preprocessing script, no baselines in the repo | `prepare_data.py`, `baselines.py`, `compare.py` |

## Next steps

1. Give rare anomalies room in the representation: a much larger codebook (512–1024) or continuous
   I-JEPA-style targets with a variance regulariser instead of quantisation.
2. Add a masked-patch objective inside the window, so the encoder must encode local patch content.
3. Move the label horizon to 24h (3x more positives, the usual Azure PdM setting), for all models at once.
4. Always report PR-AUC and the from-scratch ablation: they are what make the pre-training's real
   contribution visible.
