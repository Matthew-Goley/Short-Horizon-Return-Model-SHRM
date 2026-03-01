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

        # classification head
        self.final_dropout = nn.Dropout(dropout)

        # dual head
        self.mu_head = nn.Linear(d_model, 1)      # predicted return
        self.log_var_head = nn.Linear(d_model, 1) # predicted uncertainty
    
    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)

        x = self.input_proj(x)  # (batch, seq, d_model)

        x = self.transformer(x)  # (batch, seq, d_model)

        # take representation of last timestep
        x = x[:, -1, :]  # (batch, d_model)
        x = self.final_dropout(x)

        mu = self.mu_head(x)
        log_var = self.log_var_head(x)

        return mu, log_var
    
if __name__ == "__main__":
    model = TransformerClassifier(
        feature_dim=4,
        d_model=32,
        num_heads=4,
        num_layers=2
    )

    dummy = torch.randn(40000, 24, 4)
    mu, log_var = model(dummy)

    print(f"mu shape: {mu.shape}")
    print(f"log_var shape: {log_var.shape}")