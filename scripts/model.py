import torch
import torch.nn as nn

class TransformerClassifier(nn.Module):
    def __init__(self, feature_dim, d_model, num_heads, num_layers, dropout=0.1):
        super().__init__()

        # project 2 -> d_model
        self.input_proj = nn.Linear(feature_dim, d_model)

        # add dropout before final claassifier

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True # important for some reason
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.final_dropout = nn.Dropout(dropout)
        self.mu_head = nn.Linear(d_model, 1)

        # dedicated sigma path: volat_z(2), realized_vol_norm(7), hl_range_norm(8)
        self.sigma_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)

        vol_features = x[:, -1, :][:, [2, 7, 8]] # (batch, 3)

        # mu path: full transformer
        x_enc = self.input_proj(x)
        x_enc = self.transformer(x_enc)
        x_last = self.final_dropout(x_enc[:, -1, :])
        mu = self.mu_head(x_last)

        # sigma path: independent of transformer
        log_var = self.sigma_mlp(vol_features)
        log_var = torch.clamp(log_var, min=-6, max=2)

        return mu, log_var
    
if __name__ == "__main__":
    model = TransformerClassifier(
        feature_dim=9,
        d_model=32,
        num_heads=4,
        num_layers=2
    )

    dummy = torch.randn(16, 24, 9)
    mu, log_var = model(dummy)

    print(f"mu shape: {mu.shape}")
    print(f"log_var shape: {log_var.shape}")