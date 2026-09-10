"""
SC-JEPA pre-training (encoder + soft codebook + fine/coarse predictors + decoder).

Changes with respect to the original version:
  * no per-window instance normalisation: all features are standardised once in the
    Dataset with the training scaler, exactly as in the downstream stage;
  * quantizer temperature 0.1 (was 1.0 -> near-uniform codes, trivial JEPA targets);
  * codebook / commitment losses computed in the normalised space, batch-entropy weight x200,
    dead-code reset every 500 steps (without it the codebook collapsed to ~5 codes at epoch 4);
  * 24h past / 24h future -> 4 patches of 6h (was 12h / 12h -> 2 patches);
  * collapse monitors logged every epoch (codebook perplexity, std of h, mean max prob);
  * validation loss uses a fixed reconstruction weight so epochs are comparable;
  * both the online and the EMA encoder are saved; clip_grad_norm_ (in-place version).
"""
import argparse
import copy
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import PredictiveMaintenanceDataset
from data.utils import coarse_scale_to_patch, set_seed, update_ema
from models.decoder import Decoder
from models.encoder import Encoder
from models.predictor import CoarsePredictor, TransformerPredictor
from models.quantizer import Quantizer
from utils.losses import (codebook_perplexity, entropy_losses, kl_loss_coarse,
                          kl_loss_fine, mse_alignment_loss, vq_losses)

ap = argparse.ArgumentParser()
ap.add_argument("--data_dir", default="dataset")
ap.add_argument("--ckpt_dir", default="checkpoints")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=6)
ap.add_argument("--batch_size", type=int, default=256)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--stride", type=int, default=1, help="stride between training windows")
ap.add_argument("--quant_temp", type=float, default=0.1)
ap.add_argument("--wandb", action="store_true")
args = ap.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
set_seed(42)

WINDOW_SIZE = 48          # 24h past -> 24h future
PATCH_LEN = 6
NUM_PATCHES = (WINDOW_SIZE // 2) // PATCH_LEN   # 4
LATENT_DIM = 64
NUM_CODES = 64
CNN_H_DIM = 64
NHEAD = 2
NUM_TRANS_LAYERS = 2
EMA_DECAY = 0.996

KL_FINE_WEIGHT = 1.0
KL_COARSE_WEIGHT = 0.5
MSE_WEIGHT = 0.1
BETA = 0.25
COMMITMENT_WEIGHT = 0.1
ENTROPY_SAMPLE_WEIGHT = 0.01   # 0.1 saturated the softmax -> dead codes
ENTROPY_BATCH_WEIGHT = 1.0
PRED_TEMP = 0.8
RECON_WEIGHT_START = 0.5
RECON_WEIGHT_END = 0.1
DEAD_CODE_RESET_EVERY = 500    # steps; codes with EMA usage < 0.1/K are re-seeded from encoder outputs

print("[INFO] Loading dataset...")
train_ds = PredictiveMaintenanceDataset(os.path.join(args.data_dir, "train.csv"), mode="pretrain",
                                        window_size=WINDOW_SIZE, patch_len=PATCH_LEN, stride=args.stride)
val_ds = PredictiveMaintenanceDataset(os.path.join(args.data_dir, "val.csv"), mode="pretrain",
                                      window_size=WINDOW_SIZE, patch_len=PATCH_LEN, scaler=train_ds.scaler)
IN_CHANNELS = train_ds.in_channels
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                          num_workers=4, pin_memory=True, persistent_workers=True)
# the validation loader is shuffled (fixed seed): consecutive windows of one machine are
# near-duplicates, and the batch-entropy term / perplexity would be meaningless on such batches
val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4,
                        pin_memory=True, persistent_workers=True, generator=torch.Generator().manual_seed(0))
print(f"[INFO] Train windows: {len(train_ds)}  Val windows: {len(val_ds)}  channels: {IN_CHANNELS}  patches: {NUM_PATCHES}")

encoder = Encoder(num_patches=NUM_PATCHES, patch_len=PATCH_LEN, latent_dim=LATENT_DIM, cnn_h_dim=CNN_H_DIM,
                  trans_nhead=NHEAD, trans_num_layers=NUM_TRANS_LAYERS, in_channels=IN_CHANNELS).to(DEVICE)
quantizer = Quantizer(num_codes=NUM_CODES, embedding_dim=LATENT_DIM, temperature=args.quant_temp).to(DEVICE)
predictor = TransformerPredictor(num_codes=NUM_CODES, nhead=NHEAD, num_layers=NUM_TRANS_LAYERS, hidden_dim=128,
                                 num_patches=NUM_PATCHES, latent_dim=LATENT_DIM).to(DEVICE)
coarse_predictor = CoarsePredictor(num_codes=NUM_CODES, nhead=NHEAD, num_layers=NUM_TRANS_LAYERS, hidden_dim=128,
                                   num_patches=NUM_PATCHES, latent_dim=LATENT_DIM).to(DEVICE)
decoder = Decoder(latent_dim=LATENT_DIM, out_channels=IN_CHANNELS, patch_len=PATCH_LEN).to(DEVICE)

encoder_tgt = copy.deepcopy(encoder).eval()
quantizer_tgt = copy.deepcopy(quantizer).eval()
for p in list(encoder_tgt.parameters()) + list(quantizer_tgt.parameters()):
    p.requires_grad = False

trainable = (list(encoder.parameters()) + list(quantizer.parameters()) + list(predictor.parameters())
             + list(coarse_predictor.parameters()) + list(decoder.parameters()))
optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

os.makedirs(args.ckpt_dir, exist_ok=True)
best_model_path = os.path.join(args.ckpt_dir, "encoder.pth")
best_val_loss = float("inf")
patience_counter = 0

if args.wandb:
    import wandb
    wandb.init(project="SC-JEPA", name="scjepa-pretraining",
               config={"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
                       "window_size": WINDOW_SIZE, "ema_decay": EMA_DECAY, "quant_temp": args.quant_temp})


def forward_losses(x_past, x_future, recon_weight):
    """All loss terms for one batch. Inputs are already normalised by the Dataset."""
    x_past, x_future = x_past.to(DEVICE, non_blocking=True), x_future.to(DEVICE, non_blocking=True)
    x_future_coarse = coarse_scale_to_patch(x_future, add_patch_dim=True)  # (B, 1, L, C)

    h_past = encoder(x_past)                       # (B, N, D)
    p_past, z_q_past = quantizer(h_past)           # (B, N, K), (B, N, D)

    x_recon = decoder(z_q_past)                    # (B, N, C, L)
    loss_recon = F.mse_loss(x_recon, x_past.permute(0, 1, 3, 2))

    with torch.no_grad():
        p_future, z_q_future = quantizer_tgt(encoder_tgt(x_future))
        p_future_c, _ = quantizer_tgt(encoder_tgt(x_future_coarse))

    logits_pred, z_pred = predictor(p_past)
    loss_kl = kl_loss_fine(logits_pred, p_future, PRED_TEMP)
    logits_coarse, _ = coarse_predictor(p_past)
    loss_kl_c = kl_loss_coarse(logits_coarse, p_future_c, PRED_TEMP)
    loss_mse = mse_alignment_loss(z_pred, z_q_future)
    loss_q, loss_commit = vq_losses(h_past, z_q_past)
    loss_ent_s, loss_ent_b = entropy_losses(p_past)

    loss = (KL_FINE_WEIGHT * loss_kl + KL_COARSE_WEIGHT * loss_kl_c + MSE_WEIGHT * loss_mse
            + BETA * loss_q + COMMITMENT_WEIGHT * loss_commit
            + ENTROPY_SAMPLE_WEIGHT * loss_ent_s + ENTROPY_BATCH_WEIGHT * loss_ent_b
            + recon_weight * loss_recon)
    terms = {"loss": loss.item(), "kl_fine": loss_kl.item(), "kl_coarse": loss_kl_c.item(),
             "mse_align": loss_mse.item(), "vq": loss_q.item(), "commit": loss_commit.item(),
             "ent_sample": loss_ent_s.item(), "ent_batch": loss_ent_b.item(), "recon": loss_recon.item(),
             "perplexity": codebook_perplexity(p_past).item(),
             "max_prob": p_past.max(-1).values.mean().item(),
             "h_std": h_past.detach().reshape(-1, LATENT_DIM).std(0).mean().item()}
    return loss, terms, p_past.detach(), h_past.detach()


def accumulate(agg, terms):
    for k, v in terms.items():
        agg[k] = agg.get(k, 0.0) + v


total_steps = len(train_loader) * args.epochs
global_step = 0
code_usage = torch.full((NUM_CODES,), 1.0 / NUM_CODES, device=DEVICE)  # EMA of the average assignment
print("[INFO] Starting pre-training...")
for epoch in range(1, args.epochs + 1):
    for m in (encoder, quantizer, predictor, coarse_predictor, decoder):
        m.train()
    train_agg = {}
    for x_past, x_future in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
        global_step += 1
        recon_weight = RECON_WEIGHT_START - (RECON_WEIGHT_START - RECON_WEIGHT_END) * global_step / total_steps
        loss, terms, p_past, h_past = forward_losses(x_past, x_future, recon_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()
        update_ema(encoder, encoder_tgt, EMA_DECAY)
        update_ema(quantizer, quantizer_tgt, EMA_DECAY)

        # dead-code reset (standard VQ remedy): a code that receives almost no assignment mass
        # gets no useful gradient and never recovers, so it is re-seeded with a random encoder output
        with torch.no_grad():
            code_usage.mul_(0.99).add_(p_past.mean((0, 1)), alpha=0.01)
            if global_step % DEAD_CODE_RESET_EVERY == 0:
                dead = (code_usage < 0.1 / NUM_CODES).nonzero().flatten()
                if len(dead) > 0:
                    h_flat = F.normalize(h_past.reshape(-1, LATENT_DIM), dim=-1)
                    seeds = h_flat[torch.randint(0, h_flat.shape[0], (len(dead),), device=DEVICE)]
                    quantizer.embedding.weight.data[dead] = seeds
                    quantizer_tgt.embedding.weight.data[dead] = seeds
                    code_usage[dead] = 1.0 / NUM_CODES
                    terms["code_resets"] = float(len(dead))
        accumulate(train_agg, terms)
    train_agg = {k: v / len(train_loader) for k, v in train_agg.items()}

    for m in (encoder, quantizer, predictor, coarse_predictor, decoder):
        m.eval()
    val_agg = {}
    with torch.no_grad():
        for x_past, x_future in val_loader:
            _, terms, _, _ = forward_losses(x_past, x_future, RECON_WEIGHT_END)
            accumulate(val_agg, terms)
    val_agg = {k: v / len(val_loader) for k, v in val_agg.items()}

    print(f"[EPOCH {epoch}/{args.epochs}] train {train_agg['loss']:.4f} | val {val_agg['loss']:.4f} | "
          f"kl {val_agg['kl_fine']:.3f} kl_c {val_agg['kl_coarse']:.3f} recon {val_agg['recon']:.3f} | "
          f"perplexity {val_agg['perplexity']:.1f}/{NUM_CODES} max_prob {val_agg['max_prob']:.2f} h_std {val_agg['h_std']:.3f} | "
          f"lr {optimizer.param_groups[0]['lr']:.1e} | code resets {train_agg.get('code_resets', 0.0) * len(train_loader):.0f}")
    if args.wandb:
        wandb.log({"pretrain/epoch": epoch, **{f"pretrain/train_{k}": v for k, v in train_agg.items()},
                   **{f"pretrain/val_{k}": v for k, v in val_agg.items()}})

    scheduler.step(val_agg["loss"])
    if val_agg["loss"] < best_val_loss:
        best_val_loss = val_agg["loss"]
        patience_counter = 0
        torch.save({"online": encoder.state_dict(), "ema": encoder_tgt.state_dict(),
                    "quantizer": quantizer.state_dict(),
                    "config": {"num_patches": NUM_PATCHES, "patch_len": PATCH_LEN, "latent_dim": LATENT_DIM,
                               "cnn_h_dim": CNN_H_DIM, "nhead": NHEAD, "num_layers": NUM_TRANS_LAYERS,
                               "in_channels": IN_CHANNELS}}, best_model_path)
        print(f"  --> improved, saved to {best_model_path}")
    else:
        patience_counter += 1
        print(f"  --> no improvement ({patience_counter}/{args.patience})")
        if patience_counter >= args.patience:
            print(f"[EARLY STOPPING] epoch {epoch}")
            break

if args.wandb:
    wandb.finish()
print("[INFO] Pre-training complete!")
