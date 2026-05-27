"""
tcn_arch.py — Temporal Convolutional Network for cognitive load classification.

Input  : (batch, 2, window_len)  — 2 channels: [EDA, BVP] at 4 Hz
Output : (batch, 2)              — logits for [baseline, stress]

Architecture overview
---------------------
  4 stacked residual TCN blocks with exponentially growing dilation (1, 2, 4, 8).
  Each block applies two causal dilated convolutions + BatchNorm + ReLU + Dropout,
  with a 1×1 skip connection that projects when channel dimensions change.
  Global average pooling collapses the time dimension, then a small MLP head
  produces the final class logits.

Why causal convolutions?
  At inference time the model processes a sliding window over a live stream.
  Causal padding ensures the prediction at time t only sees t and earlier —
  no future leakage, consistent behaviour between training and deployment.
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrize import remove_parametrizations


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _CausalConv1d(nn.Module):
    """
    Dilated 1-D convolution with left-only padding so the output at position t
    depends only on inputs at positions ≤ t (causal).
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation          # left-pad only
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=self.pad, dilation=dilation
        )
        nn.init.kaiming_normal_(self.conv.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        # Remove the right-side padding introduced by symmetric padding
        return out[:, :, : -self.pad] if self.pad > 0 else out


class _ResBlock(nn.Module):
    """
    TCN residual block:
      CausalConv → BN → ReLU → Dropout →
      CausalConv → BN → ReLU → Dropout →
      + skip connection (1×1 conv if dimensions differ)
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        self.conv1 = _CausalConv1d(in_ch,  out_ch, kernel_size, dilation)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = _CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout(p=dropout)
        # 1×1 projection for residual when channels change
        self.skip  = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.drop(self.relu(self.bn1(self.conv1(x))))
        h = self.drop(self.relu(self.bn2(self.conv2(h))))
        return self.relu(h + self.skip(x))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class TCN(nn.Module):
    """
    Temporal Convolutional Network for binary cognitive-load classification.

    Parameters
    ----------
    input_channels : int
        Number of input signal channels (default 2 = EDA + BVP).
    num_classes : int
        Number of output classes (default 2 = baseline / stress).
    channel_sizes : list[int]
        Feature-map widths for each residual block. Dilation for block i = 2^i.
    kernel_size : int
        Convolution kernel size (same across all blocks).
    dropout : float
        Dropout probability applied after each activation.
    """

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 2,
        channel_sizes: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        channel_sizes = channel_sizes or [32, 64, 64, 128]

        blocks = []
        for i, out_ch in enumerate(channel_sizes):
            in_ch = input_channels if i == 0 else channel_sizes[i - 1]
            blocks.append(
                _ResBlock(in_ch, out_ch, kernel_size, dilation=2 ** i, dropout=dropout)
            )
        self.backbone = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # (batch, C, T) → (batch, C, 1)
            nn.Flatten(),              # → (batch, C)
            nn.Linear(channel_sizes[-1], 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )

        # Stored so inference code can reconstruct the model without guessing
        self.config = dict(
            input_channels=input_channels,
            num_classes=num_classes,
            channel_sizes=channel_sizes,
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (batch, channels, time)

        Returns
        -------
        torch.Tensor, shape (batch, num_classes)  — raw logits
        """
        return self.head(self.backbone(x))

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience wrapper: returns class probabilities via softmax."""
        return torch.softmax(self.forward(x), dim=-1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices (argmax over logits)."""
        return self.forward(x).argmax(dim=-1)