#!/usr/bin/env python
"""
Extended VAE training script for MNIST with multiple latent dimensions.
- Architecture: encoder/decoder with 500 hidden units and ReLU activations.
- Tests latent dims of [2, 10, 20, 100].
- Records and plots train/test loss curves.
- Saves:
    * Loss curves for each latent dim in individual subplots.
    * Final reconstruction images after last epoch.
    * Generated samples from latent space.
    * Latent-space visualizations (PCA/2D scatter colored by digit).
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Create directories
os.makedirs('results_vae', exist_ok=True)
os.makedirs('results_vae/latent_space', exist_ok=True)
os.makedirs('results_vae/reconstructions', exist_ok=True)
os.makedirs('results_vae/samples', exist_ok=True)

# VAE model with parameterized latent dimension
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.fc1 = nn.Linear(784, 500)
        self.fc21 = nn.Linear(500, latent_dim)  # mu
        self.fc22 = nn.Linear(500, latent_dim)  # logvar
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
        recon = self.decode(z)
        return recon, mu, logvar

# Loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training for one epoch
def train_one_epoch(model, optimizer, loader, device):
    model.train()
    train_loss = 0
    for data, _ in loader:
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss / len(loader.dataset)

# Testing
def test_one_epoch(model, loader, device, latent_dim, epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data, labels) in enumerate(loader):
            data = data.to(device)
            recon, mu, logvar = model(data)
            test_loss += loss_function(recon, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n], recon.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), f'results_vae/reconstructions/reconstruction_ld{latent_dim}_epoch{epoch}.png', nrow=n)
    return test_loss / len(loader.dataset)

# Visualize latent space (PCA for >2 dims)
@torch.no_grad()
def visualize_latent_space(model, loader, device, latent_dim):
    model.eval()
    zs, ys = [], []
    for data, labels in loader:
        data = data.to(device)
        mu, _ = model.encode(data.view(-1, 784))
        zs.append(mu.cpu())
        ys.append(labels)
    zs = torch.cat(zs).numpy()
    ys = torch.cat(ys).numpy()
    if latent_dim > 2:
        zs_2d = PCA(n_components=2).fit_transform(zs)
    else:
        zs_2d = zs
    plt.figure(figsize=(6,6))
    scatter = plt.scatter(zs_2d[:,0], zs_2d[:,1], c=ys, cmap='tab10', alpha=0.7, s=5)
    plt.colorbar(scatter, ticks=range(10))
    plt.title(f'Latent space (dim={latent_dim})')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(f'results_vae/latent_space/latent_space_ld{latent_dim}.png')
    plt.close()

# Main experiment loop
def main():
    if torch.cuda.is_available():
        try:
            # quick check for CUDA ECC or other GPU issues
            _ = torch.tensor([0], device='cuda')
            device = torch.device("cuda")
        except RuntimeError as e:
            print(f"WARNING: CUDA unavailable ({e}). Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    batch_size, epochs = 128, 50
    transform = transforms.Compose([transforms.ToTensor()])

    train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform),
                              batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform),
                             batch_size=batch_size, shuffle=False)

    latent_dims = [2, 10, 20, 100]
    losses = {ld: {'train': [], 'test': []} for ld in latent_dims}

    for ld in latent_dims:
        print(f"===== Training VAE with latent_dim={ld} =====")
        model = VAE(latent_dim=ld).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, optimizer, train_loader, device)
            test_loss = test_one_epoch(model, test_loader, device, ld, epoch)
            losses[ld]['train'].append(train_loss)
            losses[ld]['test'].append(test_loss)
            print(f"LD={ld} Epoch {epoch}: Train={train_loss:.4f}, Test={test_loss:.4f}")

        visualize_latent_space(model, test_loader, device, ld)

        with torch.no_grad():
            sample_n = 32
            sample = torch.randn(sample_n, ld).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(sample_n, 1, 28, 28), f'results_vae/samples/sample_ld{ld}.png')

    # Plot loss curves in separate subplots
    n = len(latent_dims)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows))
    axes = axes.flatten()
    for ax, ld in zip(axes, latent_dims):
        ax.plot(losses[ld]['train'], label='Train')
        ax.plot(losses[ld]['test'], linestyle='--', label='Test')
        ax.set_title(f'Latent Dim = {ld}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
    # Remove any unused subplots
    for ax in axes[n:]:
        fig.delaxes(ax)
    fig.tight_layout()
    fig.savefig('results_vae/loss_curves.png')
    plt.close()

if __name__ == "__main__":
    main()
