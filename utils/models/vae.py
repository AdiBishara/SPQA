import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, channels, strides, kernel_size=3):
        super(Encoder, self).__init__()
        layers = []
        for i, (c, s) in enumerate(zip(channels, strides)):
            layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, c, kernel_size, stride=s, padding=1),
                    nn.BatchNorm3d(c),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(c, c, kernel_size, stride=1, padding=1),
                    nn.BatchNorm3d(c),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_channels = c
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, out_channels, channels, strides, kernel_size=3, deep_feature_shape=None):
        super(Decoder, self).__init__()
        self.deep_feature_shape = deep_feature_shape
        layers = []
        # Reverse channels and strides
        rev_channels = list(reversed(channels))
        rev_strides = list(reversed(strides))

        # We start from the bottleneck channel count
        in_c = rev_channels[0]

        for i, (c, s) in enumerate(zip(rev_channels[1:] + [out_channels], rev_strides)):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_c, c, kernel_size, stride=s, padding=1, output_padding=s - 1),
                    nn.BatchNorm3d(c),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv3d(c, c, kernel_size, stride=1, padding=1),
                    nn.BatchNorm3d(c) if i < len(rev_channels) else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True) if i < len(rev_channels) else nn.Identity(),
                )
            )
            in_c = c

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class VAE3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, image_size=[256, 256, 256], latent_dim=256):
        super(VAE3D, self).__init__()

        # Standard U-Net style channel progression
        self.channels = [32, 64, 128, 256, 512]
        self.strides = [2, 2, 2, 2, 2]  # 5 downsamples = 2^5 = 32x reduction

        self.encoder_net = Encoder(in_channels, self.channels, self.strides)

        # Calculate size at bottleneck
        # 256 / 32 = 8x8x8 spatial size
        self.bottleneck_size = [s // 32 for s in image_size]
        self.last_channel = self.channels[-1]  # 512

        # --- THE MEMORY FIX ---
        # Instead of flattening 8x8x8 (262k features), we pool to 2x2x2 (4k features)
        # This reduces parameter count by 64x
        self.pool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.flat_features = self.last_channel * 4 * 4 * 4  # 512 * 8 = 4096

        self.fc_mu = nn.Linear(self.flat_features, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_features, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.flat_features)
        self.decoder_net = Decoder(out_channels, self.channels, self.strides, deep_feature_shape=self.bottleneck_size)

    def encode(self, x):
        h = self.encoder_net(x)
        # Pool before flattening to save VRAM
        h = self.pool(h)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        # Unflatten back to 2x2x2
        h = h.view(-1, self.last_channel, 4, 4, 4)
        # Upsample back to bottleneck size (8x8x8) before decoding
        h = nn.functional.interpolate(h, size=tuple(self.bottleneck_size), mode='nearest')
        return self.decoder_net(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar