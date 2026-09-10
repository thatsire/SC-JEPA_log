import os
import copy
import torch
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader

from data.utils import (
    set_seed,
    instance_normalize,
    reverse_instance_normalize,
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
NUM_CONT_COLS = 5  
LATENT_DIM = 64     
NUM_CODES = 64      
CNN_H_DIM = 64          
NHEAD = 2           
NUM_TRANS_LAYERS = 2

NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
EMA_DECAY = 0.996 

KL_FINE_WEIGHT = 1.0
KL_COARSE_WEIGHT = 0.5
MSE_WEIGHT = 0.1
BETA = 0.5
COMMITMENT_WEIGHT = 0.25
ENTROPY_SAMPLE_WEIGHT = 0.001
ENTROPY_BATCH_WEIGHT = 0.005
PRED_TEMP = 0.8
RECON_WEIGHT_START = 0.5
RECON_WEIGHT_END = 0.1
BATCH_SIZE = 256

print("[INFO] Loading dataset...")
DATA_DIR = "/home/jovyan/workspace/SC-JEPA/data"
train_csv = os.path.join(DATA_DIR, "train.csv")
val_csv = os.path.join(DATA_DIR, "val.csv")

train_ds = PredictiveMaintenanceDataset(train_csv, window_size=WINDOW_SIZE, mode='pretrain')
val_ds = PredictiveMaintenanceDataset(val_csv, window_size=WINDOW_SIZE, mode='pretrain', scaler=train_ds.scaler)

x_sample, _ = train_ds[0]
IN_CHANNELS = x_sample.shape[1]

train_loader_pretrain = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader_pretrain = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
print(f"[INFO] Train batches: {len(train_loader_pretrain)}, Val batches: {len(val_loader_pretrain)}")

print("[INFO] Starting model initialization...")
encoder = Encoder(
    num_patches=NUM_PATCHES, patch_len=PATCH_LEN, latent_dim=LATENT_DIM, 
    cnn_h_dim=CNN_H_DIM, trans_nhead=NHEAD, trans_num_layers=NUM_TRANS_LAYERS, in_channels=IN_CHANNELS
).to(DEVICE)

quantizer = Quantizer(num_codes=NUM_CODES, embedding_dim=LATENT_DIM).to(DEVICE)

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

for p in encoder_tgt.parameters(): p.requires_grad = False
for p in quantizer_tgt.parameters(): p.requires_grad = False

optimizer = optim.Adam(
    list(encoder.parameters()) + list(quantizer.parameters()) + 
    list(predictor.parameters()) + list(coarse_predictor.parameters()) + list(decoder.parameters()),
    lr=LEARNING_RATE,
)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

CHECKPOINT_DIR = "/home/jovyan/workspace/SC-JEPA/checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
best_val_loss = float('inf')
best_model_path = os.path.join(CHECKPOINT_DIR, "encoder.pth")
PATIENCE = 10
patience_counter = 0

wandb.init(
    project="SC-JEPA",
    entity="irene-barbagallo03-universit-degli-studi-di-catania",
    name="scjepa-pretraining",
    config={
        "epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "window_size": WINDOW_SIZE,
        "ema_decay": EMA_DECAY
    }
)

print("[INFO] Starting pre-training...")

total_steps = len(train_loader_pretrain) * NUM_EPOCHS
global_step = 0

for epoch in range(NUM_EPOCHS):
    encoder.train()
    quantizer.train()
    predictor.train()
    coarse_predictor.train()
    decoder.train()

    total_train_loss = 0.0
    batch_count = 0
    
    for x_past, x_future in tqdm(train_loader_pretrain, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False):
        
        B = x_past.shape[0]
        x_past = x_past.view(B, NUM_PATCHES, PATCH_LEN, IN_CHANNELS)
        x_future = x_future.view(B, NUM_PATCHES, PATCH_LEN, IN_CHANNELS)

        global_step += 1
        progress = global_step / total_steps
        recon_weight = RECON_WEIGHT_START - (RECON_WEIGHT_START - RECON_WEIGHT_END) * progress
        
        x_past_norm, mean, std = instance_normalize(x_past)
        x_future_norm, _, _ = instance_normalize(x_future)
        x_future_coarse = coarse_scale_to_patch(x_future, add_patch_dim=True)

        # 1. Slice continuous vs categorical (dim=-1 is the feature dimension)
        x_past_cont, x_past_cat = x_past[:, :, :, :NUM_CONT_COLS], x_past[:, :, :, NUM_CONT_COLS:]
        x_future_cont, x_future_cat = x_future[:, :, :, :NUM_CONT_COLS], x_future[:, :, :, NUM_CONT_COLS:]
        
        # 2. Normalization
        x_past_cont_norm, mean, std = instance_normalize(x_past_cont)
        x_future_cont_norm, _, _ = instance_normalize(x_future_cont)
        
        # 3. Concatenate and move to device
        x_past_norm = torch.cat([x_past_cont_norm, x_past_cat], dim=-1).to(DEVICE)
        x_future_norm = torch.cat([x_future_cont_norm, x_future_cat], dim=-1).to(DEVICE)
        mean, std = mean.to(DEVICE), std.to(DEVICE)
        
        # 4. Handle coarse future normalization safely
        x_future_coarse = coarse_scale_to_patch(x_future, add_patch_dim=True)
        x_fc_cont, x_fc_cat = x_future_coarse[:, :, :, :NUM_CONT_COLS], x_future_coarse[:, :, :, NUM_CONT_COLS:]
        x_fc_cont_norm, _, _ = instance_normalize(x_fc_cont)
        x_future_coarse_norm = torch.cat([x_fc_cont_norm, x_fc_cat], dim=-1).to(DEVICE)
        
        h_past = encoder(x_past_norm)
        p_past, z_q_past = quantizer(h_past)

        x_recon = decoder(z_q_past) # Shape: (B, NUM_PATCHES, IN_CHANNELS, PATCH_LEN)
        
        # 5. Reverse normalization
        x_recon_cont = x_recon[:, :, :NUM_CONT_COLS, :]
        x_recon_cat = x_recon[:, :, NUM_CONT_COLS:, :]
        x_recon_cont_rev = reverse_instance_normalize(x_recon_cont, mean, std)
        
        x_recon = torch.cat([x_recon_cont_rev, x_recon_cat], dim=2)
        x_target = x_past.permute(0, 1, 3, 2).to(DEVICE)
        
        loss_recon = torch.nn.functional.mse_loss(x_recon, x_target)

        logits_pred, z_pred = predictor(p_past)
        h_future = encoder_tgt(x_future_norm)
        p_future, z_q_future = quantizer_tgt(h_future)
        loss_kl = kl_loss_fine(logits_pred, p_future, PRED_TEMP)

        logits_coarse, _ = coarse_predictor(p_past)
        h_future_c = encoder_tgt(x_future_coarse_norm)
        p_future_c, _ = quantizer_tgt(h_future_c)
        loss_kl_c = kl_loss_coarse(logits_coarse, p_future_c, PRED_TEMP)

        loss_mse = mse_alignment_loss(z_pred, z_q_future)
        loss_q, loss_commit = vq_losses(h_past, z_q_past)
        loss_ent_s, loss_ent_b = entropy_losses(p_past)

        loss = (KL_FINE_WEIGHT * loss_kl + KL_COARSE_WEIGHT * loss_kl_c + 
                MSE_WEIGHT * loss_mse + BETA * loss_q + 
                COMMITMENT_WEIGHT * loss_commit + ENTROPY_SAMPLE_WEIGHT * loss_ent_s + 
                ENTROPY_BATCH_WEIGHT * loss_ent_b + recon_weight * loss_recon)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(
            list(encoder.parameters()) + list(quantizer.parameters()) +
            list(predictor.parameters()) + list(coarse_predictor.parameters()) + list(decoder.parameters()),
            max_norm=1.0
        )
        optimizer.step()

        update_ema(encoder, encoder_tgt, EMA_DECAY)
        update_ema(quantizer, quantizer_tgt, EMA_DECAY)
        
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader_pretrain)

    # Validation loop
    encoder.eval()
    quantizer.eval()
    predictor.eval()
    coarse_predictor.eval()
    decoder.eval()
    
    total_val_loss = 0.0
    
    with torch.no_grad():
        for x_past, x_future in val_loader_pretrain:
            B = x_past.shape[0]
            x_past = x_past.view(B, NUM_PATCHES, PATCH_LEN, IN_CHANNELS)
            x_future = x_future.view(B, NUM_PATCHES, PATCH_LEN, IN_CHANNELS)

            # 1. Slice continuous vs categorical (dim=-1 is the feature dimension)
            x_past_cont, x_past_cat = x_past[:, :, :, :NUM_CONT_COLS], x_past[:, :, :, NUM_CONT_COLS:]
            x_future_cont, x_future_cat = x_future[:, :, :, :NUM_CONT_COLS], x_future[:, :, :, NUM_CONT_COLS:]
            
            # 2. Normalize ONLY continuous features
            x_past_cont_norm, mean, std = instance_normalize(x_past_cont)
            x_future_cont_norm, _, _ = instance_normalize(x_future_cont)
            
            # 3. Concatenate and move to device
            x_past_norm = torch.cat([x_past_cont_norm, x_past_cat], dim=-1).to(DEVICE)
            x_future_norm = torch.cat([x_future_cont_norm, x_future_cat], dim=-1).to(DEVICE)
            mean, std = mean.to(DEVICE), std.to(DEVICE)
            
            # 4. Handle coarse future normalization safely
            x_future_coarse = coarse_scale_to_patch(x_future, add_patch_dim=True)
            x_fc_cont, x_fc_cat = x_future_coarse[:, :, :, :NUM_CONT_COLS], x_future_coarse[:, :, :, NUM_CONT_COLS:]
            x_fc_cont_norm, _, _ = instance_normalize(x_fc_cont)
            x_future_coarse_norm = torch.cat([x_fc_cont_norm, x_fc_cat], dim=-1).to(DEVICE)

            h_past = encoder(x_past_norm)
            p_past, z_q_past = quantizer(h_past)
            x_recon = decoder(z_q_past)
            
            x_recon_cont = x_recon[:, :, :NUM_CONT_COLS, :]
            x_recon_cat = x_recon[:, :, NUM_CONT_COLS:, :]
            x_recon_cont_rev = reverse_instance_normalize(x_recon_cont, mean, std)
            
            x_recon = torch.cat([x_recon_cont_rev, x_recon_cat], dim=2)
            
            x_target = x_past.permute(0, 1, 3, 2).to(DEVICE)
            
            loss_recon = torch.nn.functional.mse_loss(x_recon, x_target)

            logits_pred, z_pred = predictor(p_past)
            h_future = encoder_tgt(x_future_norm)
            p_future, z_q_future = quantizer_tgt(h_future)
            loss_kl = kl_loss_fine(logits_pred, p_future, PRED_TEMP)

            logits_coarse, _ = coarse_predictor(p_past)
            
            h_future_c = encoder_tgt(x_future_coarse_norm)
            p_future_c, _ = quantizer_tgt(h_future_c)
            loss_kl_c = kl_loss_coarse(logits_coarse, p_future_c, PRED_TEMP)

            loss_mse = mse_alignment_loss(z_pred, z_q_future)
            loss_q, loss_commit = vq_losses(h_past, z_q_past)
            loss_ent_s, loss_ent_b = entropy_losses(p_past)

            val_loss = (KL_FINE_WEIGHT * loss_kl + KL_COARSE_WEIGHT * loss_kl_c + 
                        MSE_WEIGHT * loss_mse + BETA * loss_q + 
                        COMMITMENT_WEIGHT * loss_commit + ENTROPY_SAMPLE_WEIGHT * loss_ent_s + 
                        ENTROPY_BATCH_WEIGHT * loss_ent_b + recon_weight * loss_recon)
            
            total_val_loss += val_loss.item()
            
    avg_val_loss = total_val_loss / len(val_loader_pretrain)
    
    wandb.log({
        "pretrain/epoch": epoch + 1,
        "pretrain/train_loss": avg_train_loss,
        "pretrain/val_loss": avg_val_loss
    })
    
    print(f"[EPOCH {epoch+1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    scheduler.step(avg_val_loss)
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        print(f"  --> Model improved! Saving weights in {best_model_path}")
        torch.save(encoder.state_dict(), best_model_path) 
        
    else:
        patience_counter += 1
        print(f"  --> No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        if patience_counter >= PATIENCE:
            print(f"\n[EARLY STOPPING] Stopped at epoch {epoch+1}. The validation loss has not improved for {PATIENCE} epochs.")
            break

wandb.finish()
print("\n[INFO] Pre-training complete!")