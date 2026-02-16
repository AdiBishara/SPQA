import torch
import torch.nn as nn


class UpsampleBlock(nn.Module):
    """
    Replaces ConvTranspose with Upsample + Conv to eliminate checkerboard artifacts.
    """

    def __init__(self, in_ch, out_ch):
        super(UpsampleBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),  # Stable norm for small batches
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(4, out_ch),
            nn.LeakyReLU(0.2, inplace=False)
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(4, channels)
        )
        self.relu = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class VAE3D(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, image_size=[256, 256, 256], latent_dim=2048):
        super(VAE3D, self).__init__()

        # --- ENCODER (Standard) ---
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, 3, 2, 1),
            nn.GroupNorm(4, 16), nn.LeakyReLU(0.2), ResidualBlock3D(16)
        )
        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, 3, 2, 1),
            nn.GroupNorm(4, 32), nn.LeakyReLU(0.2), ResidualBlock3D(32)
        )
        self.enc3 = nn.Sequential(
            nn.Conv3d(32, 64, 3, 2, 1),
            nn.GroupNorm(8, 64), nn.LeakyReLU(0.2), ResidualBlock3D(64)
        )
        self.enc4 = nn.Sequential(
            nn.Conv3d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128), nn.LeakyReLU(0.2), ResidualBlock3D(128)
        )
        self.enc5 = nn.Sequential(
            nn.Conv3d(128, 256, 3, 2, 1),
            nn.GroupNorm(8, 256), nn.LeakyReLU(0.2), ResidualBlock3D(256)
        )

        # --- BOTTLENECK ---
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # --- DENSE SPATIAL PROJECTOR (The Fix) ---
        # Instead of broadcasting, we project directly to a 4x4x4 volume with 128 channels
        # This gives the model a "Chunk of Clay" that already has shape info
        self.spatial_base_dim = 4
        self.spatial_base_ch = 128
        self.decoder_projection = nn.Linear(
            latent_dim,
            self.spatial_base_ch * self.spatial_base_dim ** 3
        )

        # --- DECODER (Upsample-based) ---
        # We need 6 upsamples to go from 4x4x4 -> 256x256x256
        self.dec6 = UpsampleBlock(128, 128)  # 4 -> 8
        self.dec5 = UpsampleBlock(128, 64)  # 8 -> 16
        self.dec4 = UpsampleBlock(64, 64)  # 16 -> 32
        self.dec3 = UpsampleBlock(64, 32)  # 32 -> 64
        self.dec2 = UpsampleBlock(32, 16)  # 64 -> 128
        self.dec1 = UpsampleBlock(16, 16)  # 128 -> 256

        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.orthogonal_(m.weight)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        # Clamp inputs to prevent outliers
        x = torch.clamp(x, -5, 5)

        # Encode
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)

        # Bottleneck
        pooled = self.pool(x5).view(x5.shape[0], -1)
        mu = self.fc_mu(pooled)
        logvar = torch.clamp(self.fc_logvar(pooled), -10, 10)
        z = self.reparameterize(mu, logvar)

        # Decode (Dense Projection)
        # 1. Project latent vector to a massive flat vector
        z_spatial = self.decoder_projection(z)
        # 2. Reshape into a 3D volume (Batch, 128, 4, 4, 4)
        z_vol = z_spatial.view(-1, self.spatial_base_ch, self.spatial_base_dim, self.spatial_base_dim,
                               self.spatial_base_dim)

        # 3. Upsample path
        d6 = self.dec6(z_vol)  # -> 8
        d5 = self.dec5(d6)  # -> 16
        d4 = self.dec4(d5)  # -> 32
        d3 = self.dec3(d4)  # -> 64
        d2 = self.dec2(d3)  # -> 128
        d1 = self.dec1(d2)  # -> 256

        return self.final_conv(d1), mu, logvar