import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictionHead(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(PredictionHead, self).__init__()

        # FC1: 4096 → 1024
        self.fc1 = nn.Linear(4096, 1024)
        self.dropout1 = nn.Dropout(dropout_rate)

        # FC2: 1024 → 256
        self.fc2 = nn.Linear(1024, 256)
        self.dropout2 = nn.Dropout(dropout_rate)

        # FC3: 256 → 256 (residual block)
        self.fc3 = nn.Linear(256, 256)
        self.layer_norm = nn.LayerNorm(256)

        # Final output layer: 256 → 1
        self.fc4 = nn.Linear(256, 1)

        # GELU activation
        self.activation = nn.GELU()

    def forward(self, x_emb: torch.tensor, y_emb: torch.tensor) -> torch.tensor:
        """
        x_emb: Tensor of shape (batch_size, 1024) — LASER embedding of reference
        y_emb: Tensor of shape (batch_size, 1024) — LASER embedding of candidate
        """

        # Construct composite input: [x ; y ; |x - y| ; x * y]
        abs_diff = torch.abs(x_emb - y_emb)
        elementwise_product = x_emb * y_emb
        composite_input = torch.cat(
            [x_emb, y_emb, abs_diff, elementwise_product], dim=1
        )  # shape: (batch_size, 4096)

        # FC1: 4096 → 1024
        x = self.fc1(composite_input)
        x = self.activation(x)
        x = self.dropout1(x)

        # FC2: 1024 → 256
        x = self.fc2(x)
        x = self.activation(x)
        x = self.dropout2(x)

        # FC3 (residual block): 256 → 256
        residual = x
        x = self.fc3(x)
        x = x + residual  # Residual connection
        x = self.layer_norm(x)

        # FC4: 256 → 1 → sigmoid
        x = self.fc4(x)
        score = torch.sigmoid(x)  # Final score in [0, 1]

        return score
