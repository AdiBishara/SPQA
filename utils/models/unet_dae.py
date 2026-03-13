import torch
import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet

class UNetDAE(nn.Module):
    """DAE built on UNet architecture; maps 2-channel input (MRI+Corrupt Mask) to restored mask."""
    def __init__(self,
                 in_channels=2,
                 out_channels=1,
                 channels=[16, 32, 64, 128, 256],
                 strides=[2, 2, 2, 2],
                 num_res_units=2,
                 kernel_size=3,
                 norm="BATCH",
                 dropout=0.0,
                 spatial_dims=3):
        super(UNetDAE, self).__init__()

        self.model = MonaiUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            kernel_size=kernel_size,
            norm=norm,
            dropout=dropout
        )

    def forward(self, x):
        # x shape: [B, 2, D, H, W] (Image + Corrupted mask)
        recon = self.model(x)
        
        # We return recon (raw logits for BCEWithLogitsLoss under autocast), 
        # and two `None` values to represent mu and logvar.
        return recon, None, None
