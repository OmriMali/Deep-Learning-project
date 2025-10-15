"""
Generate HS dataset using a trained SHS-GAN model
- Loads trained generator
- Generates 600 HS samples from RGB dataset
- Saves each HS cube as .pt in OUTPUT_DIR
"""

import os, random, torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# --------------------
# Import your modules
# --------------------
from SHS_GAN import RGB2HSGenerator, HS2RGB, RGBFlatFolder  # assumes training script is named shs_gan_train.py

# --------------------
# Config
# --------------------
CHECKPOINT_PATH = "./checkpoints/shs_gan_epoch10.pth"
RGB_DATASET_DIR = "./datasets/rgb"
OUTPUT_DIR = "./generated_hs"
BANDS = 31
IMAGE_SIZE = 128
BATCH_SIZE = 4
NUM_SAMPLES = 600
DEVICE = torch.device("xpu" if torch.xpu.is_available() else "cpu")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Load H matrix (same as training)
# --------------------
x = torch.linspace(0, 1, BANDS)
H_r = torch.clamp(1 - torch.abs(x - 0.8) / 0.8, 0, 1)
H_g = torch.clamp(1 - torch.abs(x - 0.5) / 0.5, 0, 1)
H_b = torch.clamp(1 - torch.abs(x - 0.2) / 0.2, 0, 1)
H_mat = torch.stack([H_r, H_g, H_b], dim=0).to(DEVICE)
hs2rgb = HS2RGB(H_mat).to(DEVICE)

# --------------------
# Load Generator
# --------------------
print("[LOAD] Loading trained generator...")
ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
G = RGB2HSGenerator(out_bands=BANDS, base_ch=64)
G.load_state_dict(ckpt["G_state"])
G = G.to(DEVICE).eval()
print(f"[INFO] Loaded checkpoint from {CHECKPOINT_PATH}")

# --------------------
# Load RGB dataset
# --------------------
print("[DATA] Loading RGB dataset...")
rgb_ds = RGBFlatFolder(RGB_DATASET_DIR, image_size=IMAGE_SIZE)
rgb_loader = DataLoader(rgb_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
print(f"[INFO] Found {len(rgb_ds)} RGB samples")

# --------------------
# Generate HS dataset
# --------------------
print(f"[GEN] Generating up to {NUM_SAMPLES} HS samples...")
all_rgb_batches = []
all_hs_batches = []
count = 0

with torch.no_grad():
    for rgb in rgb_loader:
        rgb = rgb.to(DEVICE)
        hs_fake = G(rgb)
        all_rgb_batches.append(rgb.cpu())
        all_hs_batches.append(hs_fake.cpu())

        for i in range(hs_fake.size(0)):
            if count >= NUM_SAMPLES:
                break
            save_path = os.path.join(OUTPUT_DIR, f"hs_sample_{count:04d}.pt")
            torch.save(hs_fake[i].cpu(), save_path)
            count += 1
            print(f"[SAVE] Saved HS sample {count}/{NUM_SAMPLES} -> {save_path}")

        if count >= NUM_SAMPLES:
            break

print(f"[DONE] Generated {count} HS samples in {OUTPUT_DIR}")
