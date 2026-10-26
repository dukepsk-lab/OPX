"""1D-CNN + BiLSTM deep learning model adapter with EV-weighted loss in PyTorch."""

from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from opx.econ import _check


class CNNLSTMNet(nn.Module):
    """1D-CNN feature extractor followed by BiLSTM temporal encoder."""

    def __init__(self, in_features: int, conv_channels: int = 32, lstm_hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv1d(1, conv_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.act1 = nn.GELU()

        self.conv2 = nn.Conv1d(conv_channels, conv_channels * 2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_channels * 2)
        self.act2 = nn.GELU()

        self.lstm = nn.LSTM(
            input_size=conv_channels * 2,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.drop = nn.Dropout(0.2)
        self.head = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_features) -> (batch, 1, in_features)
        b, f = x.shape
        x_1d = x.unsqueeze(1)
        c = self.act1(self.bn1(self.conv1(x_1d)))
        c = self.act2(self.bn2(self.conv2(c)))  # (batch, channels, in_features)

        # Transpose for LSTM: (batch, in_features, channels)
        seq = c.permute(0, 2, 1)
        lstm_out, _ = self.lstm(seq)  # (batch, in_features, lstm_hidden * 2)

        # Global average pool across features sequence
        pooled = lstm_out.mean(dim=1)
        logits = self.head(self.drop(pooled)).squeeze(-1)
        return logits


def ev_bce_loss(logits: torch.Tensor, targets: torch.Tensor, r: float) -> torch.Tensor:
    """EV cost-weighted binary cross entropy loss."""
    w = r * targets + (1.0 - targets)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (w * bce).mean()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def fit_nn(
    X: np.ndarray,
    y: np.ndarray,
    r: float,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | None = None,
) -> CNNLSTMNet:
    """Train CNNLSTMNet on given feature matrix and labels."""
    _check(r)
    if device is None:
        device = get_device()

    n_features = X.shape[1]
    model = CNNLSTMNet(in_features=n_features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = ev_bce_loss(logits, by, r=r)
            loss.backward()
            optimizer.step()

    return model


def predict_nn_prob(
    model: CNNLSTMNet,
    X: np.ndarray,
    r: float,
    batch_size: int = 1024,
    device: torch.device | None = None,
) -> np.ndarray:
    """Predict uncalibrated win probabilities from the neural network raw margin."""
    if device is None:
        device = get_device()

    model.eval()
    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=False)

    logits_list = []
    with torch.no_grad():
        for (bx,) in loader:
            bx = bx.to(device)
            out = model(bx)
            logits_list.append(out.cpu().numpy())

    raw = np.concatenate(logits_list) if logits_list else np.array([])
    q = 1.0 / (1.0 + np.exp(-np.clip(raw, -30.0, 30.0)))
    denom = q + r * (1.0 - q)
    return q / denom
