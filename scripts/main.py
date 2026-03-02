import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pipeline import CCOMPUTEALL
from dataset import MarketDataset
from model import TransformerClassifier # returns mu and log_var

def gaussian_nll(y: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    # y, mu, log_var: (B,)
    # NLL up to additive constant:
    # 0.5 * [ (y-mu)^2 / exp(log_var) + log_var ]
    return 0.5 * ((y- mu) ** 2 / torch.exp(log_var) + log_var).mean()

def norm_train_stats(X_train: np.ndarray):
    # X_train: (N, L, F)
    flat = X_train.reshape(-1, X_train.shape[-1]) # (N*L, F)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    return mean, std

def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / (std + 1e-8)

def train(
    X_train, y_class_train, y_ret_train,
    X_val, y_class_val, y_ret_val,
    epochs: int = 25,
    batch_size: int = 32,
    lr: float = 5e-4,
    grad_clip: float = 1.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # Normalize using training stats
    mean, std = norm_train_stats(X_train)
    X_train = apply_norm(X_train, mean, std)
    X_val = apply_norm(X_val, mean, std)

    # return std for confidence scaling
    return_std = float(np.std(y_ret_train))
    print(f"return_std (train): {return_std}")

    train_dataset = MarketDataset(X_train, y_class_train, y_ret_train)
    val_dataset = MarketDataset(X_val, y_class_val, y_ret_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    feature_dim = X_train.shape[-1]

    model = TransformerClassifier(
        feature_dim=feature_dim,
        d_model=32,
        num_heads=4,
        num_layers=2,
        dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parameters: {total_params:,}")
    print(f"Trinable Parameters: {trainable_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_loss_sum = 0.0
        train_total = 0

        train_mu_sum = 0.0
        train_sigma_sum = 0.0

        for X_batch, y_class_batch, y_ret_batch in train_loader:
            X_batch = X_batch.to(device).float()
            y_ret_batch = y_ret_batch.to(device).float().view(-1)

            optimizer.zero_grad()

            mu, log_var = model(X_batch)
            mu = mu.view(-1).float()
            log_var = log_var.view(-1).float()

            sigma = torch.exp(0.5 * log_var)

            train_mu_sum += mu.abs().sum().item()
            train_sigma_sum += sigma.mean().item() * y_ret_batch.size(0)

            loss = gaussian_nll(y_ret_batch, mu, log_var)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            bs = y_ret_batch.size(0)
            train_loss_sum += loss.item() * bs
            train_total += bs

        avg_train_loss = train_loss_sum / max(train_total, 1)
        avg_train_mu = train_mu_sum / max(train_total, 1)
        avg_train_sigma = train_sigma_sum / max(train_total, 1)

        # validate

        model.eval()
        val_loss_sum = 0.0
        val_total = 0

        val_mu_sum = 0.0
        val_sigma_sum = 0.0
        val_conf_sum = 0.0
        val_pred_pos = 0
        val_snr_sum = 0

        with torch.no_grad():
            for X_batch, y_class_batch, y_ret_batch in val_loader:
                X_batch = X_batch.to(device).float()
                y_ret_batch = y_ret_batch.to(device).float().view(-1)

                mu, log_var = model(X_batch)
                mu = mu.view(-1).float()
                log_var = log_var.view(-1).float()

                sigma = torch.exp(0.5 * log_var)
                snr = torch.abs(mu) / (sigma + 1e-8)
                val_snr_sum += snr.mean().item() * y_ret_batch.size(0)
                confidence = 1 / (1 + sigma / return_std)

                val_mu_sum += mu.abs().sum().item()
                val_sigma_sum += sigma.mean().item() * y_ret_batch.size(0)
                val_conf_sum += confidence.mean().item() * y_ret_batch.size(0)
                val_pred_pos += (mu > 0).sum().item()

                loss = gaussian_nll(y_ret_batch, mu, log_var)

                bs = y_ret_batch.size(0)
                val_loss_sum += loss.item() * bs
                val_total += bs

        avg_val_loss = val_loss_sum / max(val_total, 1)
        avg_val_mu = val_mu_sum / max(val_total, 1)
        avg_val_sigma = val_sigma_sum / max(val_total, 1)
        avg_val_conf = val_conf_sum / max(val_total, 1)
        val_pred_rate = val_pred_pos / max(val_total, 1)
        avg_val_snr = val_snr_sum / max(val_total, 1)

        # Edge Score
        # 1. Generalization score (penalize if val worse than train)
        gen_gap = avg_train_loss - avg_val_loss
        gen_score = max(min(gen_gap * 50, 20), 0)  # cap at 20 pts

        # 2. Signal strength score (SNR based)
        snr_score = max(min(avg_val_snr * 40, 30), 0)  # cap at 30 pts

        # 3. Confidence improvement (σ vs return_std)
        vol_ratio = avg_val_sigma / return_std
        vol_score = max(min((1 - vol_ratio) * 50, 20), 0)  # up to 20 pts

        # 4. Directional balance (penalize collapse)
        direction_score = 20 * (1 - abs(val_pred_rate - 0.5) * 2)

        # Combine
        edge_score = gen_score + snr_score + vol_score + direction_score

        # Clamp 0–100
        edge_score = max(min(edge_score, 100), 0)

        print(
            f"Epoch {epoch:03d} | "
            f"Train NLL: {avg_train_loss:.6f} | "
            f"Val NLL: {avg_val_loss:.6f} | "
            f"Val |μ|: {avg_val_mu:.5f} | "
            f"Val σ: {avg_val_sigma:.5f} | "
            f"Val Conf: {avg_val_conf:.3f} | "
            f"Val pred+ rate: {val_pred_rate:.3f}"
            f"Val SNR: {avg_val_snr:.3f} | "
            f"EDGE SCORE: {edge_score:.1f}/100"
        )

    return model, mean, std, return_std

if __name__ == "__main__":
    X_train, y_class_train, y_ret_train, X_val, y_class_val, y_ret_val = CCOMPUTEALL(
        window=12, seq_len=24, len_shift=1, sector="tech"
    )

    train(
        X_train, y_class_train, y_ret_train,
        X_val, y_class_val, y_ret_val,
        epochs=10
    )