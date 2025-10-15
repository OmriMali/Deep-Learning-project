import os
import math
import time
import numpy as np
from scipy.io import loadmat

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datetime import datetime
import csv
import pandas as pd
import matplotlib.pyplot as plt

def make_run_dir(base="runs"):
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(base, ts)
    os.makedirs(path, exist_ok=True)
    return path

def plot_curves_from_csv(csv_path, out_dir):
    df = pd.read_csv(csv_path)

    # Loss vs. Epoch (TRAIN vs TEST)
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["test_loss"],  label="test_loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss vs. Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss_vs_epoch.png")); plt.close()

    # Accuracy vs. Epoch (TRAIN vs TEST)
    plt.figure()
    plt.plot(df["epoch"], df["train_acc"], label="train_acc")
    plt.plot(df["epoch"], df["test_acc"],  label="test_acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Accuracy vs. Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "acc_vs_epoch.png")); plt.close()

    # OA / AA / Kappa vs. Epoch
    plt.figure()
    plt.plot(df["epoch"], df["OA"],    label="OA")
    plt.plot(df["epoch"], df["AA"],    label="AA")
    plt.plot(df["epoch"], df["kappa"], label="kappa")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.title("OA / AA / κ vs. Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "oa_aa_kappa.png")); plt.close()

class MetricsLogger:
    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, "metrics.csv")
        self._wrote_header = False
    def log(self, row: dict):
        write_header = not os.path.exists(self.path) or not self._wrote_header
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
                self._wrote_header = True
            w.writerow(row)


# ----------------------------- dataset ------------------------------
class IndianPinesCubes(Dataset):
    """
    X: HSI as numpy array (H, W, L) with L=200 bands
    y: label map as numpy array (H, W) with values in {0..C}, where 0 may mean 'unlabeled'
    window: spatial window size (odd). Paper uses S=5.
    normalize: 'zscore' per-band normalization
    """
    def __init__(self, X, y, window=5, ignore_label=0, normalize="zscore"):
        assert window % 2 == 1, "window must be odd"
        self.X = X.astype(np.float32)  # (H, W, L)
        self.y = y.astype(np.int64)    # (H, W)
        self.window = window
        self.pad = window // 2
        self.ignore_label = ignore_label

        # per-band normalization (recommended)
        if normalize == "zscore":
            H, W, L = self.X.shape
            flat = self.X.reshape(-1, L)
            mu = flat.mean(axis=0, keepdims=True)
            sigma = flat.std(axis=0, keepdims=True) + 1e-8
            self.X = ((flat - mu) / sigma).reshape(H, W, L)

        # pad spatially so we can take centered SxS everywhere
        self.Xp = np.pad(self.X, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), mode='reflect')
        self.yp = np.pad(self.y, ((self.pad, self.pad), (self.pad, self.pad)),
                         mode='constant', constant_values=self.ignore_label)

        # collect labeled pixels
        H, W = self.y.shape
        self.idxs = [(i, j) for i in range(H) for j in range(W) if self.y[i, j] != self.ignore_label]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i, j = self.idxs[idx]
        ip, jp = i + self.pad, j + self.pad

        # crop SxS window from padded cube (S, S, L)
        cube = self.Xp[ip - self.pad: ip + self.pad + 1,
                       jp - self.pad: jp + self.pad + 1, :]

        # Conv3d expects (C, D, H, W); treat spectral as D
        cube = np.transpose(cube, (2, 0, 1))  # (L, S, S)
        cube = cube[np.newaxis, ...]          # (1, L, S, S)

        label = int(self.y[i, j])
        return torch.from_numpy(cube), torch.tensor(label, dtype=torch.long)

# ------------------------------ model -------------------------------
class HSI3DCNN_IndianPines(nn.Module):
    """
    Matches the paper's Table 8 for Indian Pines:
      - C1: 2 kernels of size 3x3x7 (we write as (D=7,H=3,W=3))
      - C2: 8 kernels of size 3x3x3
      - FC1: 128
      - FC_out: 16 classes
      - No pooling; ReLU activations; softmax via CrossEntropyLoss
    """
    def __init__(self, in_bands=200, num_classes=16, fc_dim=128):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 2, kernel_size=(7, 3, 3), stride=1, padding=0, bias=True)  # -> (N,2,194,3,3)
        self.conv2 = nn.Conv3d(2, 8, kernel_size=(3, 3, 3), stride=1, padding=0, bias=True)  # -> (N,8,192,1,1)
        self.fc1   = nn.Linear(8 * 192, fc_dim)
        self.fc_out= nn.Linear(fc_dim, num_classes)

        # He init (good for ReLU)
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))    # (N, 2, 194, 3, 3)
        x = F.relu(self.conv2(x))    # (N, 8, 192, 1, 1)
        x = x.flatten(1)             # (N, 8*192)
        x = F.relu(self.fc1(x))      # (N, 128)
        return self.fc_out(x)        # (N, 16)

# --------------------------- train / eval ---------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for cubes, labels in loader:
        cubes  = cubes.to(device, non_blocking=True)   # (B,1,L,S,S)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(cubes)
        loss = crit(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total   += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

@torch.no_grad()
def evaluate(model, loader, device, return_preds=False):
    model.eval()
    crit = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for cubes, labels in loader:
        cubes  = cubes.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(cubes)
        loss = crit(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

        if return_preds:
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    if return_preds:
        return (total_loss / max(total,1),
                correct / max(total,1),
                np.concatenate(all_labels, axis=0),
                np.concatenate(all_preds, axis=0))
    else:
        return total_loss / max(total,1), correct / max(total,1)

# ------------------------------ metrics -----------------------------
def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        # shift to [0..C-1]; dataset labels assumed 1..C, 0 unlabeled (we don't feed 0 into model)
        cm[t-1, p-1] += 1
    return cm

def compute_metrics(cm):
    N = cm.sum()
    po = np.trace(cm) / N if N > 0 else 0.0                       # OA
    with np.errstate(divide='ignore', invalid='ignore'):
        per_class_acc = np.where(cm.sum(axis=1) > 0,
                                 np.diag(cm) / cm.sum(axis=1),
                                 0.0)
    aa = per_class_acc.mean() if len(per_class_acc) > 0 else 0.0   # AA
    # Kappa
    rows = cm.sum(axis=1)
    cols = cm.sum(axis=0)
    pe = (rows @ cols) / (N * N) if N > 0 else 0.0
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 0 else 0.0
    return po, aa, kappa, per_class_acc

# --------------------------- data utilities -------------------------
def load_indian_pines(
    x_path="data/Indian_pines_corrected.mat",
    y_path="data/Indian_pines_gt.mat",
    x_key="indian_pines_corrected",
    y_key="indian_pines_gt",
):
    X = loadmat(x_path)[x_key].astype(np.float32)   # (145,145,200)
    y = loadmat(y_path)[y_key].astype(np.int64)     # (145,145), labels 0..16 (0=unlabeled)
    assert X.ndim == 3 and y.ndim == 2, f"Bad shapes: {X.shape}, {y.shape}"
    return X, y

def make_train_test_masks(y, train_ratio=0.5, seed=42):
    rng = np.random.default_rng(seed)
    coords = np.argwhere(y != 0)
    rng.shuffle(coords)
    split = int(train_ratio * len(coords))
    train_coords = coords[:split]
    test_coords  = coords[split:]
    train_mask = np.zeros_like(y, dtype=bool)
    test_mask  = np.zeros_like(y, dtype=bool)
    train_mask[train_coords[:,0], train_coords[:,1]] = True
    test_mask [test_coords [:,0], test_coords [:,1]] = True
    return train_mask, test_mask

def build_loaders(
    X, y, train_mask, test_mask,
    window=5, batch_size_train=20, batch_size_test=64
):
    # mask unlabeled points out of each split
    y_train = np.where(train_mask, y, 0)
    y_test  = np.where(test_mask,  y, 0)
    ds_train = IndianPinesCubes(X, y_train, window=window, ignore_label=0)
    ds_test  = IndianPinesCubes(X, y_test,  window=window, ignore_label=0)
    train_loader = DataLoader(ds_train, batch_size=batch_size_train, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(ds_test,  batch_size=batch_size_test,  shuffle=False, num_workers=0)
    return train_loader, test_loader

# ---------------------------- checkpoints ---------------------------
def save_ckpt(path, epoch, model, optimizer, best_acc, iters_done):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_acc": best_acc,
        "iters_done": iters_done,
    }, path)

def load_ckpt(path, model, optimizer, map_location=None):
    chk = torch.load(path, map_location=map_location)
    model.load_state_dict(chk["model"])
    optimizer.load_state_dict(chk["optimizer"])
    return chk["epoch"], chk.get("best_acc", 0.0), chk.get("iters_done", 0)

# ------------------------------- main -------------------------------
def main(
    data_dir="data",
    batch_size_train=20,           # paper uses 20
    batch_size_test=64,
    desired_iterations=100_000,    # ~100k per paper for Indian Pines
    base_lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
    seed=42,
    window=5,
    ckpt_dir="checkpoints",
    resume=False,
):
    # device (prefer Intel XPU)
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    print("Using device:", device)

    run_dir = make_run_dir("runs")
    print("Run directory:", run_dir)
    logger = MetricsLogger(run_dir)

    # load data
    X, y = load_indian_pines(
        x_path=f"{data_dir}/Indian_pines_corrected.mat",
        y_path=f"{data_dir}/Indian_pines_gt.mat",
    )
    train_mask, test_mask = make_train_test_masks(y, train_ratio=0.5, seed=seed)
    train_loader, test_loader = build_loaders(
        X, y, train_mask, test_mask,
        window=window,
        batch_size_train=batch_size_train,
        batch_size_test=batch_size_test,
    )

    # iterations -> epochs
    n_train = len(train_loader.dataset)
    steps_per_epoch = math.ceil(n_train / batch_size_train)
    total_epochs = math.ceil(desired_iterations / max(steps_per_epoch, 1))
    print(f"Train samples: {n_train} | steps/epoch: {steps_per_epoch} | target iters: {desired_iterations} → epochs: {total_epochs}")

    # model / optimizer / scheduler
    torch.manual_seed(seed)
    model = HSI3DCNN_IndianPines(in_bands=200, num_classes=16, fc_dim=128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    # resume (optional)
    last_ckpt = os.path.join(ckpt_dir, "last.pt")
    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    start_epoch, best_acc, iters_done = 0, 0.0, 0
    if resume and os.path.exists(last_ckpt):
        start_epoch, best_acc, iters_done = load_ckpt(last_ckpt, model, optimizer, map_location=device)
        print(f"Resumed from epoch {start_epoch}, best_acc={best_acc:.4f}, iters_done={iters_done}")

    # train
    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)

        # eval with predictions to compute OA/AA/kappa
        te_loss, te_acc, y_true, y_pred = evaluate(model, test_loader, device, return_preds=True)
        cm = confusion_matrix(y_true, y_pred, num_classes=16)
        oa, aa, kappa, per_class = compute_metrics(cm)

        scheduler.step()

        print(f"[Epoch {epoch+1}/{total_epochs}] "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"test loss {te_loss:.4f} acc {te_acc:.4f} | "
              f"OA {oa:.4f} AA {aa:.4f} κ {kappa:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.5f} | "
              f"time {time.time()-t0:.1f}s")

        logger.log({
            "epoch": epoch + 1,
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "test_loss": float(te_loss),
            "test_acc": float(te_acc),
            "OA": float(oa),
            "AA": float(aa),
            "kappa": float(kappa),
            "lr": float(scheduler.get_last_lr()[0]),
            "seconds": float(time.time() - t0),
        })

        # save last + best
        iters_done += steps_per_epoch
        save_ckpt(last_ckpt, epoch+1, model, optimizer, best_acc, iters_done)
        if te_acc > best_acc:
            best_acc = te_acc
            save_ckpt(best_ckpt, epoch+1, model, optimizer, best_acc, iters_done)
            print(f"  ↳ New best acc: {best_acc:.4f} (checkpoint saved)")

    plot_curves_from_csv(os.path.join(run_dir, "metrics.csv"), run_dir)
    print("Saved plots to:", run_dir)
    print("\nTraining finished.")
    print(f"Best test accuracy: {best_acc:.4f}")
    print(f"Checkpoints saved under: {ckpt_dir}")

# ------------------------------ run it ------------------------------
if __name__ == "__main__":
    # Make sure you installed the XPU wheel:
    # pip install --upgrade pip
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu
    #
    # Place the .mat files under ./data:
    #   data/Indian_pines_corrected.mat  (key: 'indian_pines_corrected')
    #   data/Indian_pines_gt.mat         (key: 'indian_pines_gt')
    main(
        data_dir="data",
        batch_size_train=20,
        batch_size_test=64,
        desired_iterations=100_000,  # adjust if you want a quicker run
        base_lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        seed=42,
        window=5,
        ckpt_dir="checkpoints",
        resume=False,
    )