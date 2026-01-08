import torch
import numpy as np
import torch.nn as nn
from typing import Sequence, Tuple


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
            intermediate: Sequence[int] = (512, 1024),  # These now represent CHANNELS, not neurons
            **kwargs
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.features = features
        self.decoder_features = decoder_features
        self.intermediate_channels = intermediate
        self.strides = strides

        self.encoded_channels = in_channels
        if decoder_features is not None:
            decode_channel_list = list(decoder_features)
        else:
            decode_channel_list = list(features[-2::-1])

        self.strides = [strides[0] for i in features] if len(strides) == 1 else strides

        # 1. Build Encoder
        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, self.features, self.strides)

        # 2. Build Fully Convolutional Bottleneck (Replaces Linear)
        # We pass the last encoder channel count to the bottleneck
        self.intermediate = self._get_intermediate_module(self.encoded_channels, self.intermediate_channels)

        # 3. Build Decoder
        # The decoder now expects input from the last intermediate channel
        last_inter_channel = intermediate[-1] if intermediate else self.encoded_channels
        self.decode, _ = self._get_decode_module(last_inter_channel, decode_channel_list, out_channels,
                                                 self.strides[::-1])

    def forward(self, x: torch.Tensor) -> torch.FloatTensor:
        x = self.encode(x)
        # REMOVED: x = x.view(x.size(0), -1) (No more flattening)
        x = self.intermediate(x)
        # REMOVED: x = x.view(...) (No more un-flattening)
        x = self.decode(x)
        return x

    def _get_encode_module(self, in_channels: int, channels: Sequence[int], strides: Sequence[int]) -> Tuple[
        nn.Sequential, int]:
        encode = nn.Sequential()
        layer_channels = in_channels
        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, i == (len(self.features) - 1))
            encode.add_module("encode_%i" % i, layer)
            layer_channels = c
        return encode, layer_channels

    def _get_intermediate_module(self, in_channels: int, channel_list: Sequence[int]) -> nn.Sequential:
        # Replaces Linear layers with 1x1 Convolutions
        intermediate = nn.Sequential()
        layer_channels = in_channels

        for i, ch in enumerate(channel_list):
            # 1x1 Conv acts exactly like a Dense layer but preserves spatial dimensions (H, W)
            layer = nn.Conv2d(layer_channels, ch, kernel_size=1, stride=1, padding=0)
            intermediate.add_module("inter_%i" % i, layer)
            intermediate.add_module("act_%i" % i, nn.ReLU())
            layer_channels = ch

        # Add a final projection back to the size expected by the decoder
        # The decoder expects the same number of channels as the *end* of the encoder?
        # In the original code, the decoder mirrored the encoder.
        # We must ensure the last intermediate layer connects to the first decoder layer.
        return intermediate

    def _get_decode_module(self, in_channels: int, channels: Sequence[int], output_channels: int,
                           strides: Sequence[int]) -> Tuple[nn.Sequential, int]:
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

    def _get_decode_layer(self, in_channels: int, out_channels: int, stride: int, final_out_ch: int,
                          is_last: bool) -> nn.Sequential:
        layer = nn.Sequential()
        layer.add_module('upconv0',
                         nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                            output_padding=1))
        layer.add_module('act0', nn.ReLU())
        if not is_last:
            layer.add_module('conv0', nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        else:
            layer.add_module('conv0', nn.Conv2d(out_channels, final_out_ch, kernel_size=3, stride=1, padding=1))
        return layer

    def get_encoder_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)


# Keep Discriminator as is (it was already compatible)
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, num_features=16):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features, num_features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 2, num_features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_features * 4, 1, kernel_size=4, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.main(x)
        out = torch.mean(out, dim=[2, 3], keepdim=True)
        out = out.view(out.size(0), -1)
        return out