import torch
import torch.nn as nn


class RangeCNNEncoder(nn.Module):
    def __init__(self, in_channels=7, mid_channels=32, out_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(out_channels, out_channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, C, H, W]
        if not hasattr(self, '_debug_done'):
            self._debug_done = True
            print(f"[RangeCNNEncoder] input shape: {x.shape}")
            print(f"[RangeCNNEncoder] input NaN: {torch.isnan(x).any().item()}")
        feats = self.encoder(x)
        s_gate = self.spatial_gate(feats)
        feats = feats * s_gate
        c_gate = self.channel_gate(feats)
        c_gate = c_gate.unsqueeze(-1).unsqueeze(-1)
        feats = feats * c_gate
        return feats

