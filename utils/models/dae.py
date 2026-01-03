import torch
import numpy as np
import torch.nn as nn
from typing import Sequence, Tuple
from torchvision import models

class DAE(nn.Module):
    def __init__(
            self,
            spatial_dims: int = 2,
            in_channels: int = 1,
            out_channels: int = 1,
            strides: Sequence[int] = (2,),
            image_size: Sequence[int] = (256, 256), 
            features: Sequence[int] = (16, 32, 32, 32, 32),
            decoder_features: Sequence[int] = (16, 16, 16, 16, 16),
            intermediate: Sequence[int] = (512, 1024),
            **kwargs
    ):
        """
        Denoising Autoencoder for Shape Priors.
        """
        super().__init__()
        self.spatial_dims = spatial_dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.decoder_features = decoder_features
        self.intermediate = intermediate
        self.strides = strides

        self.encoded_channels = in_channels
        if decoder_features is not None:
            decode_channel_list = list(decoder_features)
        else:
            decode_channel_list = list(features[-2::-1])
            
        self.strides = [strides[0] for i in features] if len(strides) == 1 else strides
        rf = np.prod(self.strides)
        self.shape_before_flattening = (features[-1], self.image_size[0] // rf, self.image_size[1] // rf)

        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, self.features, self.strides)

        flattened_size = (self.image_size[0] // rf) * (self.image_size[1] // rf) * self.encoded_channels
        self.intermediate = self._get_intermediate_module(flattened_size, self.intermediate)
        self.decode, _ = self._get_decode_module(self.encoded_channels, decode_channel_list, out_channels,
                                                 self.strides[::-1])

    def _get_encode_module(self, in_channels: int, channels: Sequence[int], strides: Sequence[int]) -> Tuple[nn.Sequential, int]:
        encode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, i == (len(self.features) - 1))
            encode.add_module("encode_%i" % i, layer)
            layer_channels = c

        return encode, layer_channels

    def _get_intermediate_module(self, in_channels: int, num_inter_units: Sequence[int]) -> nn.Sequential:
        if len(num_inter_units) < 2:
            raise NotImplementedError('There should be at least 2 intermediate layers!')

        intermediate = nn.Sequential()
        layer_channels = in_channels

        for i, units in enumerate(num_inter_units):
            if i != len(num_inter_units) - 1:
                layer = nn.Linear(layer_channels, units)
            else:
                layer = nn.Linear(layer_channels, np.prod(self.shape_before_flattening))
            intermediate.add_module("inter_%i" % i, layer)
            intermediate.add_module("act", nn.ReLU())
            layer_channels = units

        return intermediate

    def _get_decode_module(self, in_channels: int, channels: Sequence[int], output_channels: int, strides: Sequence[int]) -> Tuple[nn.Sequential, int]:
        decode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_decode_layer(layer_channels, c, s, output_channels, i == (len(channels) - 1))
            decode.add_module("decode_%i" % i, layer)
            layer_channels = c

        return decode, layer_channels

    def _get_encode_layer(self, in_channels: int, out_channels: int, stride: int, is_last: bool) -> nn.Module:
        layer = nn.Sequential()
        layer.add_module('conv0', nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1))
        layer.add_module('act0', nn.ReLU())
        if not is_last:
            layer.add_module('conv1', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layer.add_module('act1', nn.ReLU())
        return layer

    def _get_decode_layer(self, in_channels: int, out_channels: int, stride: int, final_out_ch: int, is_last: bool) -> nn.Sequential:
        layer = nn.Sequential()
        layer.add_module('upconv0', nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=1))
        layer.add_module('act0', nn.ReLU())
        
        if not is_last:
            layer.add_module('conv0', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        else:
            
            layer.add_module('conv0', nn.Conv2d(out_channels, final_out_ch, kernel_size=3, stride=1, padding=1))

        return layer

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        x = self.encode(x)
        x = x.view(x.size(0), -1)
        x = self.intermediate(x)
        x = x.view(x.size(0), *self.shape_before_flattening)
        x = self.decode(x)
        return x

    def get_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_features=16):
        """
        A simple Conv-based discriminator that outputs a single probability.
        """
        super().__init__()
        self.main = nn.Sequential(
            # Block 1: (B, 1, 256, 256) -> (B, 16, 128, 128)
            nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: (B, 16, 128, 128) -> (B, 32, 64, 64)
            nn.Conv2d(num_features, num_features*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features*2),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: (B, 32, 64, 64) -> (B, 64, 32, 32)
            nn.Conv2d(num_features*2, num_features*4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features*4),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: (B, 64, 32, 32) -> (B, 1, 29, 29) - Output layer
            # We output a "PatchGAN" style map or we can flatten it.
            # Here we reduce to 1 channel logits.
            nn.Conv2d(num_features*4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, in_channels, H, W).
        Returns:
            (B, 1) or (B, 1, H', W') raw logits.
        """
        out = self.main(x)
        # Global Average Pooling to get a single score per image
        out = torch.mean(out, dim=[2, 3], keepdim=True)
        out = out.view(out.size(0), -1) # Flatten to (B, 1)
        return out