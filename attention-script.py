#!/usr/bin/env python
"""
Transformer‑VAE experiment on MNIST: varies the number of self‑attention heads.
- Hidden dimension near 480 (divisible by seq_len*heads).
- Architecture: FC(784→480) → SelfAttentionFC(hidden_dim=480, seq_len=20, num_heads) → latent → FC → SelfAttentionFC → FC
- Tests heads in [1,2,3,4,6,8,12].
- Records train/test loss per epoch for each head setting.
- For each head count:
    * Saves final reconstruction (8 examples) under results_heads/heads_{h}/reconstructions/recon_final.png
    * Saves final generated samples (32 images) under results_heads/heads_{h}/samples/sample_final.png
    * Saves latent‑space PCA plot under results_heads/heads_{h}/latent_space/latent_space.png
- At end, plots train/test losses in a 2×4 grid: results_heads/loss_curves_heads.png
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Base results folder
base_dir = 'results_heads_2'
os.makedirs(base_dir, exist_ok=True)

torch.manual_seed(0)

class SelfAttentionFC(nn.Module):
    def __init__(self, hidden_dim, seq_len, num_heads):
        super(SelfAttentionFC, self).__init__()
        assert hidden_dim % seq_len == 0, "hidden_dim must be divisible by seq_len"
        emb_dim = hidden_dim // seq_len
        assert emb_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.seq_len = seq_len
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim,
                                          num_heads=num_heads,
                                          batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        B, H = x.size()
        emb_dim = H // self.seq_len
        seq = x.view(B, self.seq_len, emb_dim)
        attn_out, _ = self.attn(seq, seq, seq)
        flat = attn_out.contiguous().view(B, H)
        return self.ln(x + flat)

class VAE_Attn(nn.Module):
    def __init__(self, latent_dim, attn_heads, hidden_dim=480, seq_len=20):
        super(VAE_Attn, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.fc1 = nn.Linear(784, hidden_dim)
        self.attn_enc = SelfAttentionFC(hidden_dim, seq_len, attn_heads)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)
        self.fc22 = nn.Linear(hidden_dim, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.attn_dec = SelfAttentionFC(hidden_dim, seq_len, attn_heads)
        self.fc4 = nn.Linear(hidden_dim, 784)

    def encode(self, x):
        h = F.relu(self.fc1(x.view(-1, 784)))
        h = self.attn_enc(h)
        return self.fc21(h), self.fc22(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = self.attn_dec(h)
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss function
def loss_fn(recon, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon, x.view(-1,784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Train one epoch
def train_epoch(model, optimizer, loader, device):
    model.train()
    total = 0
    for data, _ in loader:
        x = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(x)
        loss = loss_fn(recon, x, mu, logvar)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader.dataset)

# Test one epoch
def test_epoch(model, loader, device):
    model.eval()
    total = 0
    mus_list, ys_list = [], []
    with torch.no_grad():
        for data, y in loader:
            x = data.to(device)
            recon, mu, logvar = model(x)
            total += loss_fn(recon, x, mu, logvar).item()
            mus_list.append(mu.cpu().numpy())
            ys_list.append(y.numpy())
    mus = np.concatenate(mus_list, axis=0)
    ys = np.concatenate(ys_list, axis=0)
    return total / len(loader.dataset), mus, ys

# Utility functions

def save_reconstruction(model, loader, device, path):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(loader))
        x = data.to(device)
        recon, _, _ = model(x)
        n = min(x.size(0), 8)
        comp = torch.cat([x[:n], recon.view(-1,1,28,28)[:n]])
        save_image(comp.cpu(), path, nrow=n)


def save_samples(model, device, path, sample_n=32):
    z = torch.randn(sample_n, model.latent_dim, device=device)
    gen = model.decode(z).cpu()
    save_image(gen.view(sample_n,1,28,28), path, nrow=8)


def plot_latent_pca(mus, ys, path):
    pca = PCA(n_components=2)
    mus2 = pca.fit_transform(mus)
    plt.figure(figsize=(6,6))
    sc = plt.scatter(mus2[:,0], mus2[:,1], c=ys, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(sc, ticks=range(10))
    plt.title('Latent PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# Main experiment
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Data loaders
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(datasets.MNIST('./data', True, download=True, transform=transform),
                              batch_size=128, shuffle=True)
    test_loader  = DataLoader(datasets.MNIST('./data', False, transform=transform),
                              batch_size=128, shuffle=False)

    latent_dim = 2
    head_list = [1, 4, 8, 12]
    epochs = 50
    losses = {h: {'train': [], 'test': []} for h in head_list}

    for h in head_list:
        print(f'=== Heads = {h} ===')
        # Setup directories
        dir_h = os.path.join(base_dir, f'heads_{h}')
        recon_dir = os.path.join(dir_h, 'reconstructions')
        sample_dir = os.path.join(dir_h, 'samples')
        lat_dir = os.path.join(dir_h, 'latent_space')
        for d in (recon_dir, sample_dir, lat_dir): os.makedirs(d, exist_ok=True)

        model = VAE_Attn(latent_dim, attn_heads=h).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for ep in range(1, epochs+1):
            tr = train_epoch(model, optimizer, train_loader, device)
            te, mus, ys = test_epoch(model, test_loader, device)
            losses[h]['train'].append(tr)
            losses[h]['test'].append(te)
            if ep == 1 or ep % 10 == 0:
                print(f'Epoch {ep}: train={tr:.4f}, test={te:.4f}')

        # Save final outputs
        save_reconstruction(model, test_loader, device,
                            os.path.join(recon_dir, 'recon_final.png'))
        save_samples(model, device,
                     os.path.join(sample_dir, 'sample_final.png'))
        plot_latent_pca(mus, ys,
                        os.path.join(lat_dir, 'latent_space.png'))

    # Plot aggregated loss curves
    cols = 4; rows = (len(head_list) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    for ax, h in zip(axes, head_list):
        ax.plot(losses[h]['train'], label='Train')
        ax.plot(losses[h]['test'], linestyle='--', label='Test')
        ax.set_title(f'Heads = {h}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    for ax in axes[len(head_list):]: fig.delaxes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'loss_curves_heads.png'))
    plt.close()

if __name__ == '__main__':
    main()
