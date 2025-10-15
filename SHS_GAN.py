"""
Full SHS-GAN training script with HS cube preloading, verbose stage prints,
and safe DataLoader workers.
"""

from __future__ import annotations
import os, glob, time
from dataclasses import dataclass
from typing import List
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
# ----------------------------
# SSIM (single-scale)
# ----------------------------
class SSIM(nn.Module):
    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self._window = None

    def _create_window(self, channel: int, device: torch.device, dtype: torch.dtype):
        gauss = torch.tensor(
            [math.exp(-(x - self.window_size // 2) ** 2 / (2 * self.sigma ** 2)) for x in range(self.window_size)],
            device=device, dtype=dtype,
        )
        gauss = gauss / gauss.sum()
        window_2d = (gauss[:, None] @ gauss[None, :]).unsqueeze(0).unsqueeze(0)
        return window_2d.repeat(channel, 1, 1, 1)

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        (_, channel, _, _) = img1.size()
        device, dtype = img1.device, img1.dtype
        if self._window is None or self._window.size(0) != channel or self._window.device != device:
            self._window = self._create_window(channel, device, dtype)

        K1, K2 = 0.01, 0.03
        L = 2.0
        C1, C2 = (K1*L)**2, (K2*L)**2

        mu1 = F.conv2d(img1, self._window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, self._window, padding=self.window_size//2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.pow(2), mu2.pow(2), mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, self._window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, self._window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, self._window, padding=self.window_size//2, groups=channel) - mu1_mu2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq+mu2_sq+C1)*(sigma1_sq+sigma2_sq+C2))
        return ssim_map.mean()

# -----------------------------
# HS -> RGB projection
# -----------------------------
class HS2RGB(nn.Module):
    def __init__(self, H: torch.Tensor, clamp: bool = True):
        super().__init__()
        assert H.dim()==2 and H.size(0)==3
        self.register_buffer("H", H.clone().float())
        self.clamp = clamp

    def forward(self, hs: torch.Tensor) -> torch.Tensor:
        rgb = torch.einsum("blhw,cl->bchw", hs, self.H)
        return torch.clamp(rgb,-1,1) if self.clamp else rgb

# -----------------------------
# 2D/3D conv blocks
# -----------------------------
def conv_block_2d(in_ch,out_ch,k=3,s=1,p=1,use_sn=False):
    conv = nn.Conv2d(in_ch,out_ch,k,s,p)
    if use_sn: conv = torch.nn.utils.spectral_norm(conv)
    return nn.Sequential(conv, nn.LeakyReLU(0.2,inplace=True))

def up_block_2d(in_ch,out_ch,use_sn=False):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                         conv_block_2d(in_ch,out_ch,use_sn=use_sn))

def down_block_2d(in_ch,out_ch,use_sn=False):
    return nn.Sequential(conv_block_2d(in_ch,out_ch,s=2,p=1,use_sn=use_sn))

def conv_block_3d(in_ch,out_ch,k=(3,3,3),s=(1,2,2),p=(1,1,1),use_sn=False):
    conv = nn.Conv3d(in_ch,out_ch,k,s,p)
    if use_sn: conv = torch.nn.utils.spectral_norm(conv)
    return nn.Sequential(conv, nn.LeakyReLU(0.2,inplace=True))

# ---------------
# Generator
# ---------------
class RGB2HSGenerator(nn.Module):
    def __init__(self,out_bands=31,base_ch=64,use_sn=False):
        super().__init__()
        C = base_ch
        self.e1 = conv_block_2d(3,C,use_sn=use_sn)
        self.d1 = down_block_2d(C,C*2,use_sn=use_sn)
        self.d2 = down_block_2d(C*2,C*4,use_sn=use_sn)
        self.bottleneck = conv_block_2d(C*4,C*8,use_sn=use_sn)
        self.u2 = up_block_2d(C*8,C*4,use_sn=use_sn)
        self.u1 = up_block_2d(C*4 + C*2, C*2,use_sn=use_sn)
        self.out = nn.Conv2d(C*2 + C, out_bands, kernel_size=1)
        self.act_out = nn.Sigmoid()

    def forward(self,x):
        e1 = self.e1(x)
        e2 = self.d1(e1)
        e3 = self.d2(e2)
        b = self.bottleneck(e3)
        d2 = self.u2(b)
        d1 = self.u1(torch.cat([d2,e2],dim=1))
        out = self.out(torch.cat([d1,e1],dim=1))
        return self.act_out(out)

# ---------------
# Critic
# ---------------
class SpectralCritic(nn.Module):
    def __init__(self,in_bands=31,base_ch=32,use_sn=True):
        super().__init__()
        C = base_ch
        self.raw_3d = nn.Sequential(
            conv_block_3d(1,C,(5,3,3),(1,2,2),use_sn=use_sn),
            conv_block_3d(C,C*2,(3,3,3),(1,2,2),use_sn=use_sn),
            conv_block_3d(C*2,C*4,(3,3,3),(1,2,2),use_sn=use_sn),
        )
        self.fft_3d = nn.Sequential(
            conv_block_3d(1,C,(5,3,3),(1,2,2),use_sn=use_sn),
            conv_block_3d(C,C*2,(3,3,3),(1,2,2),use_sn=use_sn),
            conv_block_3d(C*2,C*4,(3,3,3),(1,2,2),use_sn=use_sn),
        )
        self.fc = nn.Sequential(
            torch.nn.utils.spectral_norm(nn.Linear(C*8,C*4)) if use_sn else nn.Linear(C*8,C*4),
            nn.LeakyReLU(0.2,inplace=True),
            torch.nn.utils.spectral_norm(nn.Linear(C*4,1)) if use_sn else nn.Linear(C*4,1),
        )

    def forward(self,hs):
        x = hs.unsqueeze(1)
        f_raw = self.raw_3d(x)
        fft_spec = torch.fft.rfft(x, dim=2)
        fft_mag = torch.abs(fft_spec)
        f_fft = self.fft_3d(fft_mag)
        v_raw = f_raw.mean(dim=[2,3,4])
        v_fft = f_fft.mean(dim=[2,3,4])
        v = torch.cat([v_raw,v_fft],dim=1)
        return self.fc(v).squeeze(1)

# ----------------------
# Losses
# ----------------------
@dataclass
class SHSGANLossWeights:
    lambda_rmse: float = 1.0
    lambda_ssim: float = 1.0
    lambda_adv: float = 1.0

class GeneratorLoss(nn.Module):
    def __init__(self, hs2rgb: HS2RGB, weights: SHSGANLossWeights):
        super().__init__()
        self.hs2rgb = hs2rgb
        self.weights = weights
        self.ssim = SSIM()

    def forward(self, rgb_in, hs_fake, d_fake):
        rgb_rec = self.hs2rgb(hs_fake)
        rmse = torch.sqrt(F.mse_loss(rgb_rec,rgb_in)+1e-8)
        ssim_loss = 1.0 - self.ssim(rgb_rec,rgb_in)
        adv = -d_fake.mean()
        return self.weights.lambda_rmse*rmse + self.weights.lambda_ssim*ssim_loss + self.weights.lambda_adv*adv

def gradient_penalty(critic, real, fake, gp_lambda=10.0):
    device = real.device
    alpha = torch.rand(real.size(0), 1, 1, 1, device=device).expand_as(real)
    interp = alpha * real + (1 - alpha) * fake
    interp.requires_grad_(True)

    pred = critic(interp)

    grads = torch.autograd.grad(
        outputs=pred,
        inputs=interp,
        grad_outputs=torch.ones_like(pred),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # safer than .view()
    grads = grads.reshape(grads.size(0), -1)

    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean() * gp_lambda
    return gp

# ----------------------
# Datasets
# ----------------------
class RGBFlatFolder(Dataset):
    def __init__(self, root_dir, image_size=256):
        exts = ["*.jpg","*.jpeg","*.png","*.ppm","*.bmp","*.tif","*.tiff"]
        files=[]
        for e in exts: files.extend(glob.glob(os.path.join(root_dir,e)))
        self.files = sorted(files)
        self.transform = transforms.Compose([
            transforms.Resize((image_size,image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3,[0.5]*3)
        ])
        if len(self.files)==0:
            print(f"[WARN] No RGB files found in {root_dir}")
        else:
            print(f"[INFO] Found {len(self.files)} RGB files (example: {self.files[0]})")

    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.transform(img)

class PaperHSIDatasetPreload(Dataset):
    def __init__(self, root_dir: str, bands: int = 31, image_size: int = 256, verbose=True):
        self.bands = bands
        self.image_size = image_size
        self.files = sorted(glob.glob(os.path.join(root_dir,"*.mat")))
        self.entries=[]
        self.verbose=verbose
        print(f"[INFO] Found {len(self.files)} .mat files in {root_dir}")

        # Stage 1: filter usable cubes
        for f in self.files:
            try:
                import h5py
                with h5py.File(f,'r') as mat:
                    if "rad" in mat:
                        arr = np.array(mat["rad"])
                        if arr.ndim==3 and arr.shape[0]==bands:
                            self.entries.append(f)
                            if verbose:
                                print(f"[STAGE 1] Usable cube: {os.path.basename(f)} shape={arr.shape}")
            except Exception as e:
                if verbose: print(f"[WARN] Could not open {os.path.basename(f)}: {e}")

        print(f"[INFO] {len(self.entries)} usable HS cubes will be preloaded")

        # Stage 2: preload & resize
        self.preloaded=[]
        for idx,f in enumerate(self.entries):
            if verbose: print(f"[STAGE 2] Loading {idx+1}/{len(self.entries)}: {os.path.basename(f)}")
            arr = self._load_and_resize(f)
            self.preloaded.append(arr)
            if verbose: print(f"[STAGE 2] Done: {arr.shape}")
        print(f"[INFO] Finished preloading {len(self.preloaded)} HS cubes")

    def _load_and_resize(self,path):
        import h5py
        arr = np.array(h5py.File(path,'r')["rad"]).astype(np.float32)
        maxv = arr.max() if arr.max()!=0 else 1.0
        arr /= maxv
        t = torch.from_numpy(arr).unsqueeze(0)
        t = F.interpolate(t, size=(self.image_size,self.image_size), mode='bilinear', align_corners=False)
        return t.squeeze(0)

    def __len__(self): return len(self.preloaded)
    def __getitem__(self, idx): return self.preloaded[idx]

# ---------------------------
# Trainer
# ---------------------------
@dataclass
class SHSGANConfig:
    bands: int=31
    image_size: int=128
    g_channels: int=64
    d_channels: int=32
    batch_size: int=4
    epochs: int=10
    n_critic: int=5
    lr_g: float=1e-5
    lr_d: float=5e-5
    lambda_rmse: float=1.0
    lambda_ssim: float=1.0
    lambda_adv: float=0.1
    use_gp: bool=True
    gp_lambda: float=10.0
    use_spectral_norm: bool=False
    checkpoint_dir: str="./checkpoints"
    print(f"config: torch.xpu.is_available(): {torch.xpu.is_available()}")
    device: str="xpu" if torch.xpu.is_available() else "cpu"

class SHSGANTrainer:
    def __init__(self,H_matrix: torch.Tensor, cfg: SHSGANConfig):
        self.cfg = cfg
        # self.device = torch.device(cfg.device)
        self.device = torch.device("xpu")
        print(f"torch.xpu.is_available(): {torch.xpu.is_available()}")
        print(f"[INFO] Using device: {self.device}")
        self.G = RGB2HSGenerator(out_bands=cfg.bands, base_ch=cfg.g_channels, use_sn=cfg.use_spectral_norm).to(self.device)
        self.D = SpectralCritic(in_bands=cfg.bands, base_ch=cfg.d_channels, use_sn=cfg.use_spectral_norm).to(self.device)
        self.hs2rgb = HS2RGB(H_matrix.to(self.device))
        self.crit = GeneratorLoss(self.hs2rgb, SHSGANLossWeights(cfg.lambda_rmse, cfg.lambda_ssim, cfg.lambda_adv)).to(self.device)
        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=cfg.lr_g, betas=(0.5,0.999))
        self.opt_D = torch.optim.RMSprop(self.D.parameters(), lr=cfg.lr_d)
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, epoch):
        path = os.path.join(self.cfg.checkpoint_dir,f"shs_gan_epoch{epoch}.pth")
        torch.save({
            "G_state": self.G.state_dict(),
            "D_state": self.D.state_dict(),
            "opt_G": self.opt_G.state_dict(),
            "opt_D": self.opt_D.state_dict(),
            "epoch": epoch
        }, path)
        print(f"[INFO] Saved checkpoint: {path}")

    def fit(self, rgb_ds, hs_ds, max_steps_per_epoch: int = 50):
        if len(rgb_ds) == 0 or len(hs_ds) == 0:
            raise RuntimeError(f"Empty dataset(s) RGB={len(rgb_ds)} HSI={len(hs_ds)}")

        G_losses = []
        D_losses = []

        for epoch in range(self.cfg.epochs):
            epoch_start = time.time()
            print(f"\n[TRAIN] Starting epoch {epoch + 1}/{self.cfg.epochs}")

            # Recreate loaders each epoch → reshuffles data
            rgb_loader = DataLoader(rgb_ds, batch_size=self.cfg.batch_size, shuffle=True,
                                    drop_last=True, num_workers=2, pin_memory=True)
            hs_loader = DataLoader(hs_ds, batch_size=self.cfg.batch_size, shuffle=True,
                                   drop_last=True, num_workers=2, pin_memory=True)
            hs_iter = iter(hs_loader)

            for step, rgb in enumerate(rgb_loader):
                rgb = rgb.to(self.device)

                # --- Train Discriminator ---
                for _ in range(self.cfg.n_critic):
                    try:
                        real_hs = next(hs_iter)
                    except StopIteration:
                        hs_iter = iter(hs_loader)
                        real_hs = next(hs_iter)
                    real_hs = real_hs.to(self.device)

                    with torch.no_grad():
                        fake_hs = self.G(rgb)

                    d_real = self.D(real_hs)
                    d_fake = self.D(fake_hs)
                    loss_D = -(d_real.mean() - d_fake.mean())

                    if self.cfg.use_gp:
                        gp = gradient_penalty(self.D, real_hs, fake_hs, self.cfg.gp_lambda)
                        loss_D += gp

                    self.opt_D.zero_grad(set_to_none=True)
                    loss_D.backward()
                    self.opt_D.step()

                # --- Train Generator ---
                fake_hs = self.G(rgb)
                d_fake = self.D(fake_hs)
                loss_G = self.crit(rgb, fake_hs, d_fake)

                self.opt_G.zero_grad(set_to_none=True)
                loss_G.backward()
                self.opt_G.step()

                # --- Track losses ---
                G_losses.append(loss_G.item())
                D_losses.append(loss_D.item())

                if step % 10 == 0:
                    print(f"[INFO] notice: Using device: {self.device}")
                    print(f"[STEP] Epoch {epoch + 1} | Step {step}/{len(rgb_loader)} "
                          f"| D: {loss_D.item():.4f} | G: {loss_G.item():.4f}")

                # Limit steps per epoch
                if step >= max_steps_per_epoch:
                    print(f"[INFO] Reached max_steps_per_epoch={max_steps_per_epoch}, breaking early")
                    break

            epoch_time = time.time() - epoch_start
            print(f"[EPOCH] {epoch + 1}/{self.cfg.epochs} finished in {epoch_time:.1f}s")
            self.save_checkpoint(epoch + 1)

        # --- Plot losses after training ---
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="Generator")
        plt.plot(D_losses, label="Discriminator")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    @torch.no_grad()
    def generate(self, rgb):
        self.G.eval()
        return self.G(rgb.to(self.device))

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    BANDS = 31
    # Example spectral response [3, L]
    x = torch.linspace(0,1,BANDS)
    H_r = torch.clamp(1 - torch.abs(x-0.8)/0.8, 0, 1)
    H_g = torch.clamp(1 - torch.abs(x-0.5)/0.5, 0, 1)
    H_b = torch.clamp(1 - torch.abs(x-0.2)/0.2, 0, 1)
    H_mat = torch.stack([H_r,H_g,H_b], dim=0)

    # Configuration
    cfg = SHSGANConfig(
        bands=BANDS,
        image_size=128,  # smaller for faster runs
        batch_size=4,
        epochs=10,
        n_critic=3,
        lr_g=1e-5,
        lr_d=5e-5,
        lambda_rmse=1.0,
        lambda_ssim=1.0,
        lambda_adv=0.1,
        use_gp=True,
        use_spectral_norm=False,
        checkpoint_dir="./checkpoints",
        device="xpu" if torch.xpu.is_available() else "cpu"
    )
    # Dataset paths
    rgb_root = "./datasets/rgb"
    hs_root = "./datasets/hs"

    # Load datasets
    print("[DATA] Loading RGB dataset...")
    rgb_ds = RGBFlatFolder(rgb_root, image_size=cfg.image_size)
    print("[DATA] Loading and preloading HS dataset...")
    hs_ds = PaperHSIDatasetPreload(hs_root, bands=cfg.bands, image_size=cfg.image_size, verbose=True)

    print(f"[DATA] RGB samples: {len(rgb_ds)} | HS samples: {len(hs_ds)}")

    # Initialize trainer
    trainer = SHSGANTrainer(H_matrix=H_mat, cfg=cfg)

    # Train
    trainer.fit(rgb_ds, hs_ds)

    # Generate sample from first 2 RGB images
    if len(rgb_ds) >= 2:
        sample_rgb = torch.stack([rgb_ds[i] for i in range(2)])
        hs_fake = trainer.generate(sample_rgb)
        print("[RESULT] Generated HS shape:", hs_fake.shape)
