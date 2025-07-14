#!/usr/bin/env python
"""
Convolutional VAE baseline for Fashion‑MNIST.

This script implements a vanilla ConvVAE (no self‑attention) on Fashion‑MNIST:
- Encoder: Conv2d → Conv2d → flatten → FC → μ/logvar
- Decoder: FC → unflatten → ConvTranspose2d → ConvTranspose2d
- Latent dimension = 50
- Loss = BCE + KLD (β=1)
- Saves per‑epoch reconstructions & 32‑sample generations
- Plots final train/test loss curve

Requirements: PyTorch, torchvision, matplotlib.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

# output directories
out_dir = 'results_fashion_conv_baseline'
os.makedirs(out_dir, exist_ok=True)
os.makedirs(os.path.join(out_dir, 'reconstructions'), exist_ok=True)
os.makedirs(os.path.join(out_dir, 'samples'), exist_ok=True)

class ConvVAE(nn.Module):
    def __init__(self, latent_dim=50):
        super(ConvVAE, self).__init__()
        self.latent_dim = latent_dim
        # Encoder
        self.enc_conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1)   # 28→14
        self.enc_conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 14→7
        self.flatten_dim = 64 * 7 * 7
        # latent heads
        self.fc_mu     = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        # Decoder
        self.fc_decode   = nn.Linear(latent_dim, self.flatten_dim)
        self.dec_deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)  # 7→14
        self.dec_deconv2 = nn.ConvTranspose2d(32,  1, kernel_size=4, stride=2, padding=1)  # 14→28

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.view(-1, self.flatten_dim)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_decode(z))
        h = h.view(-1, 64, 7, 7)
        h = F.relu(self.dec_deconv1(h))
        return torch.sigmoid(self.dec_deconv2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Loss = BCE + KLD
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epoch, model, optimizer, loader, device):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar = model(data)
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 100 == 0:
            print(f'Train Epoch {epoch} [{batch_idx*len(data)}/{len(loader.dataset)}] '
                  f'Loss: {loss.item()/len(data):.4f}')
    avg = total_loss / len(loader.dataset)
    print(f'====> Epoch {epoch} Average train loss: {avg:.4f}')
    return avg

@torch.no_grad()
def test(epoch, model, loader, device):
    model.eval()
    total_loss = 0
    for i, (data, _) in enumerate(loader):
        data = data.to(device)
        recon, mu, logvar = model(data)
        total_loss += loss_function(recon, data, mu, logvar).item()
        # save first‐batch reconstruction
        if i == 0:
            n = min(data.size(0), 8)
            comp = torch.cat([data[:n], recon[:n]])
            save_image(comp.cpu(),
                       os.path.join(out_dir, 'reconstructions', f'recon_epoch{epoch}.png'),
                       nrow=n)
    avg = total_loss / len(loader.dataset)
    print(f'====> Epoch {epoch} Test loss: {avg:.4f}')
    return avg

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # hyperparams
    batch_size    = 128
    epochs        = 40
    learning_rate = 1e-3
    latent_dim    = 50

    # data loaders
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=True,  download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(
        datasets.FashionMNIST('./data', train=False, download=True, transform=transform),
        batch_size=batch_size, shuffle=False)

    # model & optimizer
    model     = ConvVAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, test_losses = [], []
    for epoch in range(1, epochs+1):
        tr = train(epoch, model, optimizer, train_loader, device)
        te = test(epoch, model, test_loader,  device)    # <-- fixed call
        train_losses.append(tr)
        test_losses.append(te)

        # save 32 random samples each epoch
        with torch.no_grad():
            z = torch.randn(32, latent_dim, device=device)
            samples = model.decode(z).cpu()
            save_image(samples.view(32,1,28,28),
                       os.path.join(out_dir, 'samples', f'sample_epoch{epoch}.png'),
                       nrow=8)

    # plot loss curves
    plt.figure(figsize=(8,6))
    plt.plot(range(1, epochs+1), train_losses, label='Train')
    plt.plot(range(1, epochs+1), test_losses,  label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ConvVAE Baseline Loss (Fashion‑MNIST)')
    plt.legend()
    plt.savefig(os.path.join(out_dir, 'loss_plot_conv_baseline.png'))
    plt.close()

    print('Training complete. Results in:', out_dir)


if __name__ == '__main__':
    main()
