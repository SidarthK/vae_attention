#!/usr/bin/env python
"""
Expanded VAE implementation for MNIST based on the original "Auto-Encoding Variational Bayes" paper.
This script trains a VAE model, then evaluates it by testing and saving reconstructed and generated images.

Key points:
- The model is trained using the MNIST dataset.
- Reconstructed images: A comparison of original and reconstructed images to gauge performance.
- Generated images: Samples produced by decoding randomly generated latent vectors.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image

# Create a directory to save output images if it doesn't already exist.
os.makedirs('results', exist_ok=True)

# Define the Variational Autoencoder (VAE) model.
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        # Encoder layers: from input (28x28 flattened to 784) to a hidden representation.
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(784, 400)
        # Two separate layers to output the parameters for the latent space:
        # one for the mean and one for the log variance (we use 20 latent dimensions).
        self.fc21 = nn.Linear(400, latent_dim)  # Outputs the mean (mu)
        self.fc22 = nn.Linear(400, latent_dim)  # Outputs the log variance (logvar)

        # Decoder layers: from the latent space back to the hidden space, and then to reconstruct the image.
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 784)

    def encode(self, x):
        """
        Encodes the input image into latent space parameters.
        :param x: input tensor of shape [batch_size, 784]
        :return: tuple (mu, logvar)
        """
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)  # Return the mean and log variance

    def reparameterize(self, mu, logvar):
        """
        Applies the reparameterization trick: z = mu + std * epsilon.
        This allows us to backpropagate through the stochastic sampling.
        :param mu: mean of the latent Gaussian
        :param logvar: log variance of the latent Gaussian
        :return: sampled latent variable z
        """
        std = torch.exp(0.5 * logvar)  # Compute standard deviation
        eps = torch.randn_like(std)    # Sample epsilon from a standard normal distribution
        return mu + eps * std          # Return the reparameterized latent variable

    def decode(self, z):
        """
        Decodes the latent variable z back into an image.
        :param z: latent variable
        :return: reconstructed image tensor of shape [batch_size, 784]
        """
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))  # Use sigmoid to constrain output pixel values between 0 and 1

    def forward(self, x):
        """
        Forward pass through the VAE.
        :param x: input tensor
        :return: tuple (reconstructed image, mu, logvar)
        """
        mu, logvar = self.encode(x.view(-1, 784))  # Flatten input images
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def loss_function(recon_x, x, mu, logvar):
    """
    Computes the VAE loss function which is the sum of:
    - Reconstruction loss (Binary Cross Entropy) between the input and its reconstruction.
    - KL divergence loss between the learned latent distribution and a standard normal distribution.
    :param recon_x: reconstructed image
    :param x: original input image
    :param mu: latent mean from the encoder
    :param logvar: latent log variance from the encoder
    :return: total loss for the batch
    """
    # Compute reconstruction loss
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    # Compute KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train(epoch, model, optimizer, train_loader, device):
    """
    Train the VAE for one epoch.
    :param epoch: current epoch number
    :param model: VAE model
    :param optimizer: optimizer (Adam)
    :param train_loader: DataLoader for the training set
    :param device: device to run the training on (CPU or GPU)
    """
    model.train()  # Set model to training mode
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()  # Reset gradients
        recon_batch, mu, logvar = model(data)  # Forward pass
        loss = loss_function(recon_batch, data, mu, logvar)  # Compute loss
        loss.backward()  # Backpropagate
        train_loss += loss.item()
        optimizer.step()  # Update model parameters

        # Print training status every 100 batches
        if batch_idx % 100 == 0:
            print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item() / len(data):.6f}')

    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch {epoch} Average training loss: {avg_train_loss:.4f}')

def test(epoch, model, test_loader, device):
    """
    Evaluate the VAE on the test dataset.
    :param epoch: current epoch number (used for saving results)
    :param model: VAE model
    :param test_loader: DataLoader for the test set
    :param device: device to run evaluation on (CPU or GPU)
    """
    model.eval()  # Set model to evaluation mode
    test_loss = 0
    with torch.no_grad():  # Disable gradient computation for evaluation
        for i, (data, _) in enumerate(test_loader):
            data = data.to(device)
            recon, mu, logvar = model(data)
            test_loss += loss_function(recon, data, mu, logvar).item()
            # For the first batch, save a comparison of original and reconstructed images.
            if i == 0:
                n = min(data.size(0), 8)  # Number of images to display
                # 'comparison' concatenates original images with their reconstructions for visual inspection.
                comparison = torch.cat([data[:n],
                                        recon.view(-1, 1, 28, 28)[:n]])
                save_image(comparison.cpu(), f'results/reconstruction_{epoch}.png', nrow=n)
    avg_test_loss = test_loss / len(test_loader.dataset)
    print(f'====> Test set loss: {avg_test_loss:.4f}')
    return avg_test_loss

def main():
    # Check if a GPU is available; otherwise, fall back to CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    batch_size = 128
    latent_dim = 50
    epochs = 10  # You can increase the number of epochs for better performance
    learning_rate = 1e-3

    # Data loading and transformation: Convert images to tensors.
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./data', train=False, transform=transform),
        batch_size=batch_size, shuffle=True)

    # Initialize the model and optimizer.
    model = VAE(latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training and evaluation loop.
    for epoch in range(1, epochs + 1):
        train(epoch, model, optimizer, train_loader, device)
        test(epoch, model, test_loader, device)
        # Generate and save sample images by decoding random latent vectors.
        with torch.no_grad():
            # Generate 64 random latent vectors from a standard normal distribution.
            sample = torch.randn(64, latent_dim).to(device)
            sample = model.decode(sample).cpu()
            # Save generated images to the results directory.
            save_image(sample.view(64, 1, 28, 28), f'results/sample_{epoch}.png')
    print("Training and testing complete.")

if __name__ == "__main__":
    main()
