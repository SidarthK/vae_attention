#!/usr/bin/env python
"""
Beta-VAE experiment for MNIST with fixed latent dimension of 20.
- Architecture: encoder/decoder with 500 hidden units and ReLU activations.
- Varies beta in [0.1, 0.5, 1, 2, 4, 8].
- Records and plots train/test loss curves per beta in a 2×3 grid.
- Saves for each beta:
    * First-batch reconstructions at each epoch under reconstructions/.
    * 32 generated samples at final epoch under samples/.
    * Latent-space visualization (PCA to 2D colored by digit labels) under latent_space/.
- Overall loss curves saved as beta_vae/loss_curves.png
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

# Root directory for all results
top_dir = 'beta_vae_2'
os.makedirs(top_dir, exist_ok=True)

# VAE model with fixed latent dimension
torch.manual_seed(0)
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.fc1 = nn.Linear(784, 500)
        self.fc21 = nn.Linear(500, latent_dim)
        self.fc22 = nn.Linear(500, latent_dim)
        # Decoder
        self.fc3 = nn.Linear(latent_dim, 500)
        self.fc4 = nn.Linear(500, 784)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        x = x.view(-1, 784)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Beta-VAE loss
def loss_function(recon_x, x, mu, logvar, beta=1.0):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

# Training step
def train_one_epoch(model, optimizer, loader, device, beta):
    model.train()
    total = 0
    for data, _ in loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader.dataset)

# Testing step and save reconstruction
def test_one_epoch(model, loader, device, beta, recon_dir, epoch):
    model.eval()
    total = 0
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            data = data.to(device)
            recon, mu, logvar = model(data)
            total += loss_function(recon, data, mu, logvar, beta).item()
            if i == 0:
                n = min(data.size(0), 8)
                comp = torch.cat([data[:n], recon.view(-1,1,28,28)[:n]])
                save_image(comp.cpu(), os.path.join(recon_dir, f'recon_epoch{epoch}.png'), nrow=n)
    return total / len(loader.dataset)

# Latent-space PCA visualization
def visualize_latent_space(model, loader, device, latent_dir):
    model.eval()
    zs, ys = [], []
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            mu, _ = model.encode(data.view(-1,784))
            zs.append(mu.cpu().numpy())
            ys.append(labels.numpy())
    zs = np.concatenate(zs, axis=0)
    ys = np.concatenate(ys, axis=0)
    pca = PCA(n_components=2)
    zs2 = pca.fit_transform(zs)
    plt.figure(figsize=(6,6))
    scat = plt.scatter(zs2[:,0], zs2[:,1], c=ys, cmap='tab10', s=5, alpha=0.7)
    plt.colorbar(scat, ticks=range(10))
    plt.title('Latent space (beta-VAE)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(os.path.join(latent_dir, 'latent_space.png'))
    plt.close()

# Main experiment
def main():
    # Device setup
    if torch.cuda.is_available():
        try:
            _ = torch.tensor([0], device='cuda')
            device = torch.device('cuda')
        except RuntimeError:
            print('CUDA error, falling back to CPU')
            device = torch.device('cpu')
    else:
        device = torch.device('cpu')
    print('Device:', device)

    # Data loaders
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                              batch_size=128, shuffle=True)
    test_loader  = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                              batch_size=128, shuffle=False)

    latent_dim = 2
    betas = [0.1, 1, 4, 8]
    epochs = 50
    losses = {b: {'train': [], 'test': []} for b in betas}

    # Loop over betas
    for beta in betas:
        print(f'=== Beta={beta} ===')
        # Setup directories per beta
        result_dir = os.path.join(top_dir, f'beta_{beta}')
        recon_dir  = os.path.join(result_dir, 'reconstructions')
        sample_dir = os.path.join(result_dir, 'samples')
        latent_dir = os.path.join(result_dir, 'latent_space')
        for d in (recon_dir, sample_dir, latent_dir): os.makedirs(d, exist_ok=True)

        model = VAE(latent_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        # Train/test epochs
        for epoch in range(1, epochs+1):
            tr = train_one_epoch(model, optimizer, train_loader, device, beta)
            te = test_one_epoch(model, test_loader, device, beta, recon_dir, epoch)
            losses[beta]['train'].append(tr)
            losses[beta]['test'].append(te)
            if epoch==1 or epoch%10==0:
                print(f'Epoch {epoch}: Train={tr:.4f}, Test={te:.4f}')

        # Generate and save samples
        with torch.no_grad():
            z = torch.randn(32, latent_dim, device=device)
            gen = model.decode(z).cpu()
            save_image(gen.view(32,1,28,28), os.path.join(sample_dir, 'sample_final.png'), nrow=8)

        # Save latent-space plot
        visualize_latent_space(model, test_loader, device, latent_dir)

    # Plot combined loss curves
    fig, axes = plt.subplots(1,3, figsize=(15,8))
    axes = axes.flatten()
    for ax, beta in zip(axes, betas):
        ax.plot(losses[beta]['train'], label='Train')
        ax.plot(losses[beta]['test'], linestyle='--', label='Test')
        ax.set_title(f'Beta={beta}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    for ax in axes[len(betas):]: fig.delaxes(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(top_dir, 'loss_curves.png'))
    plt.close()

if __name__ == '__main__':
    main()
