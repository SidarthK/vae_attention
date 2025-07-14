#!/usr/bin/env python
"""
Conv-Attention VAE for Fashion-MNIST: varies self-attention heads.
- Encoder: Conv2d → Conv2d → flatten → FC(3136→480) → SelfAttentionFC(hidden_dim=480, seq_len=20, num_heads)
        → FC to mu/logvar (latent_dim=20)
- Decoder: FC(latent_dim→480) → SelfAttentionFC(hidden_dim=480, seq_len=20, num_heads)
        → FC(480→3136) → reshape → ConvTranspose2d → ConvTranspose2d
- Tests heads in [4,8,12], keeps latent_dim=20.
- Saves per-head:
    * Reconstructions (8 images) at final epoch: results_conv_heads/heads_{h}/reconstructions/recon_final.png
    * Generated samples (32) at final epoch: results_conv_heads/heads_{h}/samples/sample_final.png
    * Latent-space PCA: results_conv_heads/heads_{h}/latent_space/latent_space.png
- Aggregates loss curves in a 1×3 grid: results_conv_heads/loss_curves_heads.png
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

base_dir = 'results_conv_heads_20'
os.makedirs(base_dir, exist_ok=True)

torch.manual_seed(0)

class SelfAttentionFC(nn.Module):
    def __init__(self, hidden_dim, seq_len, num_heads):
        super().__init__()
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

class ConvAttnVAE(nn.Module):
    def __init__(self, latent_dim, attn_heads, hidden_dim=480, seq_len=20):
        super().__init__()
        self.latent_dim = latent_dim
        # conv encoder
        self.enc_conv1 = nn.Conv2d(1,32,4,2,1)
        self.enc_conv2 = nn.Conv2d(32,64,4,2,1)
        self.flatten_dim = 64*7*7
        # fc + attention
        self.fc1 = nn.Linear(self.flatten_dim, hidden_dim)
        self.attn_enc = SelfAttentionFC(hidden_dim, seq_len, attn_heads)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # decoder fc + attention
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.attn_dec = SelfAttentionFC(hidden_dim, seq_len, attn_heads)
        self.fc4 = nn.Linear(hidden_dim, self.flatten_dim)
        # conv transpose decoder
        self.dec_deconv1 = nn.ConvTranspose2d(64,32,4,2,1)
        self.dec_deconv2 = nn.ConvTranspose2d(32,1,4,2,1)

    def encode(self, x):
        h = F.relu(self.enc_conv1(x))
        h = F.relu(self.enc_conv2(h))
        h = h.view(-1, self.flatten_dim)
        h = F.relu(self.fc1(h))
        h = self.attn_enc(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = self.attn_dec(h)
        h = F.relu(self.fc4(h))
        h = h.view(-1,64,7,7)
        h = F.relu(self.dec_deconv1(h))
        return torch.sigmoid(self.dec_deconv2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# loss
def loss_fn(recon, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon, x, reduction='sum')
    KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE + KLD

# train/test

def train_epoch(model,opt,loader,device):
    model.train()
    total=0
    for data, _ in loader:
        x=data.to(device)
        opt.zero_grad()
        recon,mu,logvar=model(x)
        loss=loss_fn(recon,x,mu,logvar)
        loss.backward()
        opt.step()
        total+=loss.item()
    return total/len(loader.dataset)

@torch.no_grad()
def test_epoch(model,loader,device):
    model.eval()
    total=0; mus=[]; ys=[]
    for data,y in loader:
        x=data.to(device)
        recon,mu,logvar=model(x)
        total+=loss_fn(recon,x,mu,logvar).item()
        mus.append(mu.cpu().numpy()); ys.append(y.numpy())
    return total/len(loader.dataset), np.concatenate(mus), np.concatenate(ys)

# utils

def save_recon(model,loader,device,path):
    model.eval()
    with torch.no_grad():
        data,_=next(iter(loader)); x=data.to(device)
        recon,_,_=model(x)
        n=min(x.size(0),8)
        comp=torch.cat([x[:n],recon[:n]])
        save_image(comp.cpu(),path,nrow=n)

def save_sample(model,device,path,sample_n=32):
    z=torch.randn(sample_n,model.latent_dim,device=device)
    gen=model.decode(z).cpu()
    save_image(gen.view(sample_n,1,28,28),path,nrow=8)

def plot_pca(mus,ys,path):
    pca=PCA(2); z2=pca.fit_transform(mus)
    plt.figure(figsize=(6,6))
    sc=plt.scatter(z2[:,0],z2[:,1],c=ys,cmap='tab10',s=5,alpha=0.7)
    plt.colorbar(sc,ticks=range(10))
    plt.title('Latent PCA')
    plt.savefig(path);plt.close()

# main
if __name__=='__main__':
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:',device)
    transform=transforms.ToTensor()
    train_loader=DataLoader(datasets.FashionMNIST('./data',True,download=True,transform=transform),batch_size=128,shuffle=True)
    test_loader=DataLoader(datasets.FashionMNIST('./data',False,transform=transform),batch_size=128)
    latent_dim=20; head_list=[4,8,12]; epochs=50
    losses={h:{'train':[],'test':[]} for h in head_list}
    for h in head_list:
        print(f'=== Heads={h} ===')
        d=os.path.join(base_dir,f'heads_{h}')
        r,s,l=[os.path.join(d,n) for n in ['reconstructions','samples','latent_space']]
        for p in [r,s,l]:os.makedirs(p,exist_ok=True)
        model=ConvAttnVAE(latent_dim,h).to(device)
        opt=optim.Adam(model.parameters(),lr=1e-3)
        for ep in range(1,epochs+1):
            tr=train_epoch(model,opt,train_loader,device)
            te,mus,ys=test_epoch(model,test_loader,device)
            losses[h]['train'].append(tr);losses[h]['test'].append(te)
            if ep%10==0 or ep==1:print(f'E{ep}: tr={tr:.4f}, te={te:.4f}')
        save_recon(model,test_loader,device,os.path.join(r,'recon_final.png'))
        save_sample(model,device,os.path.join(s,'sample_final.png'))
        plot_pca(mus,ys,os.path.join(l,'latent_space.png'))
    # plot
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,h in zip(axes,head_list):
        ax.plot(losses[h]['train'],label='Train');ax.plot(losses[h]['test'],linestyle='--',label='Test')
        ax.set_title(f'Heads={h}');ax.legend()
    plt.tight_layout();plt.savefig(os.path.join(base_dir,'loss_curves_heads.png'))
