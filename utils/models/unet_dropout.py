from monai.networks.nets import UNet as MonaiUNet
import torch

class UNet(MonaiUNet):
    """
    Wrapper for MONAI UNet to ensure compatibility with the model loader.
    """
    def __init__(
        self,
        spatial_dims=2,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        dropout=0.2,  # Default dropout for MC uncertainty
        **kwargs
    ):
        super().__init__(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            dropout=dropout,
            **kwargs
        )