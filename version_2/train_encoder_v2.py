import os
import copy
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from data.utils import (
    set_seed,
    update_ema,
    coarse_scale_to_patch
)
from data.datasets import PredictiveMaintenanceDataset

from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import Quantizer
from models.predictor import TransformerPredictor, CoarsePredictor

from utils.losses import (
    kl_loss_fine,
    kl_loss_coarse,
    mse_alignment_loss,
    vq_losses,
    entropy_losses
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")
set_seed(42)

WINDOW_SIZE = 24  
PATCH_LEN = 6
NUM_PATCHES = (WINDOW_SIZE // 2) // PATCH_LEN
LATENT_DIM = 64     
NUM_CODES = 64      
CNN_H_DIM = 64          
NHEAD = 2           
NUM_TRANS_LAYERS = 2

NUM_EPOCHS = 30
LEARNING_RATE = 3e-4
EMA_DECAY = 0.996 

KL_FINE_WEIGHT = 1.0
KL_COARSE_WEIGHT = 0.5
MSE_WEIGHT = 0.1
BETA = 0.25
COMMITMENT_WEIGHT = 0.1
ENTROPY_SAMPLE_WEIGHT = 0.01
ENTROPY_BATCH_WEIGHT = 1.0
PRED_TEMP = 0.8
RECON_WEIGHT_START = 0.5
RECON_WEIGHT_END = 0.1
BATCH_SIZE = 256
DEAD_CODE_RESET_EVERY = 500

print("[INFO] Loading dataset...")
DATA_DIR = "/SC-JEPA/data"
train_csv = os.path.join(DATA_DIR, "train.csv")
val_csv = os.path.join(DATA_DIR, "val.csv")

train_ds = PredictiveMaintenanceDataset(train_csv, window_size=WINDOW_SIZE, mode='pretrain')
val_ds = PredictiveMaintenanceDataset(val_csv, window_size=WINDOW_SIZE, mode='pretrain', scaler=train_ds.scaler)
x_sample, _ = train_ds[0]
IN_CHANNELS = x_sample.shape[1]

train_loader_pretrain = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader_pretrain = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True, generator=torch.Generator().manual_seed(0))
print(f"[INFO] Train batches: {len(train_loader_pretrain)}, Val batches: {len(val_loader_pretrain)}")

print("[INFO] Starting model initialization...")
encoder = Encoder(
    num_patches=NUM_PATCHES, patch_len=PATCH_LEN, latent_dim=LATENT_DIM, 
    cnn_h_dim=CNN_H_DIM, trans_nhead=NHEAD, trans_num_layers=NUM_TRANS_LAYERS, in_channels=IN_CHANNELS
).to(DEVICE)

quantizer = Quantizer(num_codes=NUM_CODES, embedding_dim=LATENT_DIM, temperature=0.1).to(DEVICE)

predictor = TransformerPredictor(
    num_codes=NUM_CODES, nhead=NHEAD, num_layers=NUM_TRANS_LAYERS, 
    hidden_dim=128, num_patches=NUM_PATCHES, latent_dim=LATENT_DIM
).to(DEVICE)

coarse_predictor = CoarsePredictor(
    num_codes=NUM_CODES, nhead=NHEAD, num_layers=NUM_TRANS_LAYERS, 
    hidden_dim=128, num_patches=NUM_PATCHES, latent_dim=LATENT_DIM
).to(DEVICE)

decoder = Decoder(latent_dim=LATENT_DIM, out_channels=IN_CHANNELS, patch_len=PATCH_LEN).to(DEVICE)

print("[INFO] Preparing EMA Encoder...")
encoder_tgt = copy.deepcopy(encoder).eval()
quantizer_tgt = copy.deepcopy(quantizer).eval()

for p in list(encoder_tgt.parameters()) + list(quantizer_tgt.parameters()): 
    p.requires_grad = False

trainable = (list(encoder.parameters()) + list(quantizer.parameters()) + 
             list(predictor.parameters()) + list(coarse_predictor.parameters()) + list(decoder.parameters()))
optimizer = optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

CHECKPOINT_DIR = "/SC-JEPA/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
best_val_loss = float('inf')
best_model_path = os.path.join(CHECKPOINT_DIR, "encoder.pth")
PATIENCE = 6
patience_counter = 0

print("[INFO] Starting pre-training...")

total_steps = len(train_loader_pretrain) * NUM_EPOCHS
global_step = 0
code_usage = torch.full((NUM_CODES,), 1.0 / NUM_CODES, device=DEVICE) 

for epoch in range(NUM_EPOCHS):
    for m in (encoder, quantizer, predictor, coarse_predictor, decoder):
        m.train()

    total_train_loss = 0.0
    
    for x_past, x_future in tqdm(train_loader_pretrain, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
        
        x_past = x_past.to(DEVICE, non_blocking=True)
        x_future = x_future.to(DEVICE, non_blocking=True)

        global_step += 1
        progress = global_step / total_steps
        recon_weight = RECON_WEIGHT_START - (RECON_WEIGHT_START - RECON_WEIGHT_END) * progress
        
        x_future_coarse = coarse_scale_to_patch(x_future, add_patch_dim=True)
        
        h_past = encoder(x_past)
        p_past, z_q_past = quantizer(h_past)

        x_recon = decoder(z_q_past) 
        x_target = x_past.permute(0, 1, 3, 2)
        loss_recon = torch.nn.functional.mse_loss(x_recon, x_target)

        with torch.no_grad():
            h_future = encoder_tgt(x_future)
            p_future, z_q_future = quantizer_tgt(h_future)
            
            h_future_c = encoder_tgt(x_future_coarse)
            p_future_c, _ = quantizer_tgt(h_future_c)

        logits_pred, z_pred = predictor(p_past)
        loss_kl = kl_loss_fine(logits_pred, p_future, PRED_TEMP)

        logits_coarse, _ = coarse_predictor(p_past)
        loss_kl_c = kl_loss_coarse(logits_coarse, p_future_c, PRED_TEMP)

        loss_mse = mse_alignment_loss(z_pred, z_q_future)
        loss_q, loss_commit = vq_losses(h_past, z_q_past)
        loss_ent_s, loss_ent_b = entropy_losses(p_past)

        loss = (KL_FINE_WEIGHT * loss_kl + KL_COARSE_WEIGHT * loss_kl_c + 
                MSE_WEIGHT * loss_mse + BETA * loss_q + 
                COMMITMENT_WEIGHT * loss_commit + ENTROPY_SAMPLE_WEIGHT * loss_ent_s + 
                ENTROPY_BATCH_WEIGHT * loss_ent_b + recon_weight * loss_recon)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
        optimizer.step()

        update_ema(encoder, encoder_tgt, EMA_DECAY)
        update_ema(quantizer, quantizer_tgt, EMA_DECAY)
        
        with torch.no_grad():
            code_usage.mul_(0.99).add_(p_past.mean((0, 1)), alpha=0.01)
            if global_step % DEAD_CODE_RESET_EVERY == 0:
                dead = (code_usage < 0.1 / NUM_CODES).nonzero().flatten()
                if len(dead) > 0:
                    h_flat = torch.nn.functional.normalize(h_past.reshape(-1, LATENT_DIM), dim=-1)
                    seeds = h_flat[torch.randint(0, h_flat.shape[0], (len(dead),), device=DEVICE)]
                    quantizer.embedding.weight.data[dead] = seeds
                    quantizer_tgt.embedding.weight.data[dead] = seeds
                    code_usage[dead] = 1.0 / NUM_CODES

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader_pretrain)

    # Validation loop
    for m in (encoder, quantizer, predictor, coarse_predictor, decoder):
        m.eval()
    
    total_val_loss = 0.0
    
    with torch.no_grad():
        for x_past, x_future in val_loader_pretrain:
            x_past = x_past.to(DEVICE, non_blocking=True)
            x_future = x_future.to(DEVICE, non_blocking=True)
            x_future_coarse = coarse_scale_to_patch(x_future, add_patch_dim=True)

            h_past = encoder(x_past)
            p_past, z_q_past = quantizer(h_past)
            x_recon = decoder(z_q_past)
            x_target = x_past.permute(0, 1, 3, 2)
            loss_recon = torch.nn.functional.mse_loss(x_recon, x_target)

            h_future = encoder_tgt(x_future)
            p_future, z_q_future = quantizer_tgt(h_future)
            logits_pred, z_pred = predictor(p_past)
            loss_kl = kl_loss_fine(logits_pred, p_future, PRED_TEMP)

            h_future_c = encoder_tgt(x_future_coarse)
            p_future_c, _ = quantizer_tgt(h_future_c)
            logits_coarse, _ = coarse_predictor(p_past)
            loss_kl_c = kl_loss_coarse(logits_coarse, p_future_c, PRED_TEMP)

            loss_mse = mse_alignment_loss(z_pred, z_q_future)
            loss_q, loss_commit = vq_losses(h_past, z_q_past)
            loss_ent_s, loss_ent_b = entropy_losses(p_past)

            val_loss = (KL_FINE_WEIGHT * loss_kl + KL_COARSE_WEIGHT * loss_kl_c + 
                        MSE_WEIGHT * loss_mse + BETA * loss_q + 
                        COMMITMENT_WEIGHT * loss_commit + ENTROPY_SAMPLE_WEIGHT * loss_ent_s + 
                        ENTROPY_BATCH_WEIGHT * loss_ent_b + RECON_WEIGHT_END * loss_recon)
            
            total_val_loss += val_loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader_pretrain)
    
    print(f"[EPOCH {epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print(f"  --> Model improved! Saving weights in {best_model_path}")
        
        torch.save({
            "online": encoder.state_dict(), 
            "ema": encoder_tgt.state_dict(),
            "quantizer": quantizer.state_dict(),
            "config": {
                "num_patches": NUM_PATCHES, "patch_len": PATCH_LEN, 
                "latent_dim": LATENT_DIM, "cnn_h_dim": CNN_H_DIM, 
                "nhead": NHEAD, "num_layers": NUM_TRANS_LAYERS,
                "in_channels": IN_CHANNELS
            }
        }, best_model_path) 
        
    else:
        patience_counter += 1
        print(f"  --> No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"\n[EARLY STOPPING] Stopped at epoch {epoch+1}.")
            break

print("\n[INFO] Pre-training complete!")