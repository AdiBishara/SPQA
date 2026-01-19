import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    """
    Ensures fine anatomical details (contours) are preserved through
    skip connections, preventing the 'fuzzy blob' effect.
    """

    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        # The residual connection x + conv(x) allows fine gradients to flow
        return self.relu(x + self.conv(x))


class Encoder(nn.Module):
    def __init__(self, in_channels, channels, strides):
        super(Encoder, self).__init__()
        layers = []
        for c, s in zip(channels, strides):
            layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, c, kernel_size=3, stride=s, padding=1),
                    nn.BatchNorm3d(c),
                    nn.LeakyReLU(0.2, inplace=True),
                    ResidualBlock3D(c)
                )
            )
            in_channels = c
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, out_channels, channels, strides):
        super(Decoder, self).__init__()
        layers = []
        rev_channels = list(reversed(channels))
        rev_strides = list(reversed(strides))
        in_c = rev_channels[0]

        for i, (c, s) in enumerate(zip(rev_channels[1:] + [out_channels], rev_strides)):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose3d(in_c, c, kernel_size=3, stride=s, padding=1, output_padding=s - 1),
                    nn.BatchNorm3d(c),
                    nn.LeakyReLU(0.2, inplace=True),
                    # We only use Residual Blocks in the upsampling layers
                    ResidualBlock3D(c) if i < len(rev_channels) else nn.Identity()
                )
            )
            in_c = c
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class VAE3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, image_size=[256, 256, 256], latent_dim=2048):
        """
        image_size: (D, H, W)
        latent_dim: Set to 2048 for high-fidelity contour memory
        """
        super(VAE3D, self).__init__()

        self.channels = [32, 64, 128, 256, 512]
        self.strides = [2, 2, 2, 2, 2]  # 5 downsamples total (256 -> 8)

        self.encoder_net = Encoder(in_channels, self.channels, self.strides)

        # Bottleneck spatial size is 8x8x8 for a 256x256x256 input
        self.bottleneck_size = [s // 32 for s in image_size]
        self.last_channel = self.channels[-1]

        # Calculate flattened features (512 * 8 * 8 * 8)
        self.flat_features = self.last_channel * self.bottleneck_size[0] * self.bottleneck_size[1] * \
                             self.bottleneck_size[2]

        # Latent space heads
        self.fc_mu = nn.Linear(self.flat_features, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_features, latent_dim)

        # Decoder entry
        self.decoder_input = nn.Linear(latent_dim, self.flat_features)
        self.decoder_net = Decoder(out_channels, self.channels, self.strides)

    def encode(self, x):
        h = self.encoder_net(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # Clamp for numerical stability (prevents Exploding Gradient/NaNs)
        logvar = torch.clamp(logvar, min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_input(z)
        h = h.view(-1, self.last_channel, *self.bottleneck_size)
        return self.decoder_net(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar