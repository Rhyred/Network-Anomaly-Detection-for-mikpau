import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    """
    Implementasi Autoencoder menggunakan PyTorch.
    Arsitekturnya dibuat mirip dengan versi Keras yang asli.
    """
    def __init__(self, input_dim):
        super(AutoEncoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 8)  # Lapisan latent representation
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Linear(16, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, input_dim),
            nn.Sigmoid() # Menggunakan Sigmoid karena input di-scale antara 0 dan 1
        )
        
    def forward(self, x):
        """
        Mendefinisikan alur forward pass.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_latent_representation(self, x):
        """
        Hanya menjalankan encoder untuk mendapatkan latent representation.
        """
        return self.encoder(x)
