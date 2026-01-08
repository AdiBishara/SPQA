import torch
import torch.nn as nn
from monai.networks.nets import UNet as MonaiUNet


class UNet(nn.Module):
    def __init__(self,
                 in_channels=1,
                 out_channels=1,
                 channels=[16, 32, 64, 128, 256],
                 dropout=0.2,
                 strides=[2, 2, 2, 2],
                 spatial_dims=2):

        super(UNet, self).__init__()

        self.model = MonaiUNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=2,
            dropout=dropout
        )

        # Force Dropout (QC Requirement)
        if dropout > 0.0:
            for module in self.model.modules():
                if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                    module.p = dropout
                    module.train()

    def forward(self, x):
        # Just return the model output directly
        # It will be 512x512 because your strides are symmetric
        return self.model(x)