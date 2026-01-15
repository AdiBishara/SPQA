import torch
import torch.nn as nn


class VAE3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, image_size=[96, 96, 96], latent_dim=256):
        """
        3D Variational Autoencoder for Quality Control
        Args:
            image_size: Tuple (D, H, W) e.g., (96, 96, 96)
        """
        super(VAE3D, self).__init__()

        # --- ENCODER ---
        # We downsample 4 times. 96 -> 48 -> 24 -> 12 -> 6
        self.encoder = nn.Sequential(
            # Block 1: 96 -> 48
            nn.Conv3d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: 48 -> 24
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: 24 -> 12
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 4: 12 -> 6
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Calculate bottleneck size dynamically
        # Assuming input is divisible by 16 (2^4)
        d_feat = image_size[0] // 16
        h_feat = image_size[1] // 16
        w_feat = image_size[2] // 16

        self.flatten_size = 256 * d_feat * h_feat * w_feat
        self.reshape_dim = (256, d_feat, h_feat, w_feat)

        # --- LATENT SPACE (The "Variational" part) ---
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)

        # --- DECODER ---
        self.fc_decode = nn.Linear(latent_dim, self.flatten_size)

        self.decoder = nn.Sequential(
            # Block 4 Upsample: 6 -> 12
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),

            # Block 3 Upsample: 12 -> 24
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),

            # Block 2 Upsample: 24 -> 48
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),

            # Block 1 Upsample: 48 -> 96
            nn.ConvTranspose3d(32, out_channels, kernel_size=4, stride=2, padding=1)
            # No Sigmoid here: handled in loss function (BCEWithLogits)
        )

    def reparameterize(self, mu, logvar):
        """
        The "Reparameterization Trick":
        z = mu + sigma * epsilon
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # During inference (QC), we just use the mean
            return mu

    def forward(self, x):
        # 1. Encode
        x_enc = self.encoder(x)
        x_flat = torch.flatten(x_enc, 1)

        # 2. Latent Distribution
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        z = self.reparameterize(mu, logvar)

        # 3. Decode
        x_dec_input = self.fc_decode(z)
        x_dec_input = x_dec_input.view(-1, *self.reshape_dim)
        reconstruction = self.decoder(x_dec_input)

        return reconstruction, mu, logvar