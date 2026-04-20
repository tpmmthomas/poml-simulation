"""Tiny U-Net model for PoML simulation.

Architecture mimics a diffusion model's denoiser f_theta:
- Input: [batch, 2, 8, 8] (channel 0 = noise from seed, channel 1 = conditioning)
- Output: [batch, 1, 8, 8] (predicted noise / denoised output)

2 down-blocks, bottleneck, 2 up-blocks with skip connections.
Channels: 2 -> 8 -> 16 -> 8 -> 1.  Total ~2-5K params.
"""

import torch
import torch.nn as nn


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        skip = self.conv(x)
        down = self.pool(skip)
        return down, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class TinyUNet(nn.Module):
    """Tiny U-Net: 2→8→16→8→1, spatial 8×8."""

    def __init__(self):
        super().__init__()
        self.down1 = DownBlock(2, 8)    # 8x8 -> 4x4
        self.down2 = DownBlock(8, 16)   # 4x4 -> 2x2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )

        self.up2 = UpBlock(16, 16, 8)  # 2x2 -> 4x4
        self.up1 = UpBlock(8, 8, 8)    # 4x4 -> 8x8

        self.out_conv = nn.Conv2d(8, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1, skip1 = self.down1(x)
        d2, skip2 = self.down2(d1)

        b = self.bottleneck(d2)

        u2 = self.up2(b, skip2)
        u1 = self.up1(u2, skip1)

        return self.out_conv(u1)


def export_onnx(output_path: str = "model/network.onnx") -> None:
    """Export the TinyUNet to ONNX format."""
    import os

    import onnx

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    model = TinyUNet()
    model.eval()

    dummy_input = torch.randn(1, 2, 8, 8)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,
        opset_version=17,
    )

    # Validate
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Exported TinyUNet to {output_path} ({param_count} params)")


if __name__ == "__main__":
    export_onnx()
