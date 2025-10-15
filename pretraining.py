# pretraining.py
# -----------------------------------------------------------

# -----------------------------------------------------------

import os
import glob
import time
import csv
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ====================== Utilities ======================

def setup_seed(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def zscore_per_band_hw_last(x_hw_l: np.ndarray) -> np.ndarray:
    """
    x_hw_l: (H, W, L) -> Z-score לכל band. מחזיר float32.
    """
    H, W, L = x_hw_l.shape
    flat = x_hw_l.reshape(-1, L).astype(np.float32)
    mu = flat.mean(axis=0, keepdims=True)
    sd = flat.std(axis=0, keepdims=True) + 1e-8
    z = (flat - mu) / sd
    return z.reshape(H, W, L).astype(np.float32)


def reflect_pad_hw_last(x_hw_l: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(x_hw_l, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")


# ====================== Pretraining Data ======================

class GeneratedHSIPTDataset(Dataset):
    """
    קורא קבצי *.pt מהתיקיה. כל קובץ הוא:
      - torch.Tensor בגודל (L, H, W), או
      - dict עם מפתח 'cube' -> tensor (L, H, W).
    מחזיר:
      - patch: Tensor (1, L, S, S) לפורמט Conv3d
      - target: Tensor (L,)  — ספקטרום הפיקסל המרכזי
    """
    def __init__(self, root: str, window: int = 5, mask_ratio: float = 0.15):
        assert window % 2 == 1, "window must be odd"
        self.root = root
        self.window = window
        self.pad = window // 2
        self.mask_ratio = mask_ratio

        paths = sorted(glob.glob(os.path.join(root, "*.pt")))
        if not paths:
            raise FileNotFoundError(f"No .pt files found under '{root}'")
        self.paths = paths

        self.cubes: List[np.ndarray] = []
        self.L = None
        for p in self.paths:
            obj = torch.load(p, map_location="cpu")
            if isinstance(obj, dict):
                if "cube" in obj:
                    t = obj["cube"]
                else:
                    t = None
                    for k in ("hsi", "data", "X"):
                        if k in obj:
                            t = obj[k]
                            break
                    if t is None:
                        raise KeyError(f"{p}: dict has no 'cube' (or fallback) key")
            elif isinstance(obj, torch.Tensor):
                t = obj
            else:
                raise TypeError(f"{p}: unsupported object type {type(obj)}")

            if t.ndim != 3:
                raise ValueError(f"{p}: expected 3D tensor (L,H,W), got {tuple(t.shape)}")
            L, H, W = t.shape
            if self.L is None:
                self.L = L
            elif self.L != L:
                raise ValueError(f"All files must have same band count. Got {self.L} and {L} in {p}")

            x = t.detach().cpu().float().numpy()    # (L,H,W)
            x = np.transpose(x, (1, 2, 0))          # (H,W,L)
            x = zscore_per_band_hw_last(x)          # נרמול
            x = reflect_pad_hw_last(x, self.pad)    # ריפוד
            self.cubes.append(x)

        # אינדקס של כל מרכז חוקי בכל הקוביות (דגימה צפופה)
        self.samples: List[Tuple[int, int, int]] = []
        for k, xpad in enumerate(self.cubes):
            Hp, Wp, _ = xpad.shape
            H = Hp - 2 * self.pad
            W = Wp - 2 * self.pad
            for i in range(self.pad, self.pad + H):
                for j in range(self.pad, self.pad + W):
                    self.samples.append((k, i, j))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        k, pi, pj = self.samples[idx]
        xpad = self.cubes[k]
        p = self.pad
        patch = xpad[pi - p: pi + p + 1, pj - p: pj + p + 1, :]  # (S,S,L)
        target = xpad[pi, pj, :]                                 # (L,)

        patch = np.transpose(patch, (2, 0, 1))[None, ...].astype(np.float32)  # (1,L,S,S)

        # Random spectral masking (מייצב למידה)
        if self.mask_ratio > 0:
            L = patch.shape[1]
            m_idx = np.random.choice(L, size=max(1, int(self.mask_ratio * L)), replace=False)
            patch[:, m_idx, :, :] = 0.0

        return torch.from_numpy(patch), torch.from_numpy(target.astype(np.float32))


# ====================== Backbone + Head ======================

class HSIBackbone(nn.Module):
    """
    ה-Backbone הקונבולוציוני (Conv3d) — כמו בקוד הסופי שלך.
    קלט : (N, 1, L, S, S)
    פלט : וקטור פיצ'רים בגודל 8*(L-8) כאשר S=5 וגרעינים (7,3,3) ואז (3,3,3).
    """
    def __init__(self, in_bands: int):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 2, kernel_size=(7, 3, 3))
        self.conv2 = nn.Conv3d(2, 8, kernel_size=(3, 3, 3))
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv1.bias)
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))   # -> (N,2,L-6,3,3)
        x = F.relu(self.conv2(x))   # -> (N,8,L-8,1,1)
        x = torch.flatten(x, 1)     # -> (N, 8*(L-8))
        return x


class SpectralRegressionHead(nn.Module):
    """
    ראש רגרסיה קטן שממפה פיצ'רים לוקטור ספקטרום (L).
    """
    def __init__(self, in_features: int, out_bands: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, out_bands)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ====================== Pretraining Loop ======================

@dataclass
class PretrainConfig:
    pt_root: str = "generated_hs"
    window: int = 5
    mask_ratio: float = 0.15
    batch_size: int = 256
    epochs: int = 40
    lr: float = 1e-3
    seed: int = 42
    out_path: str = "pretrained_backbone_from_pt.pth"


def pretrain_self_supervised(cfg: PretrainConfig):
    setup_seed(cfg.seed)
    device = get_device()
    print(f"[Device] {device}")

    ds = GeneratedHSIPTDataset(cfg.pt_root, window=cfg.window, mask_ratio=cfg.mask_ratio)
    L = ds.L
    print(f"[Pretraining] Found {len(ds)} patches | Bands L={L}")
    dl = DataLoader(ds,
                    batch_size=cfg.batch_size,
                    shuffle=True,
                    num_workers=2 if device.type == "cuda" else 0,
                    pin_memory=(device.type == "cuda"))

    backbone = HSIBackbone(in_bands=L).to(device)
    head = SpectralRegressionHead(in_features=8 * (L - 8), out_bands=L).to(device)

    opt = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()), lr=cfg.lr)
    criterion = nn.MSELoss()

    for ep in range(1, cfg.epochs + 1):
        backbone.train(); head.train()
        running, n = 0.0, 0
        for xb, yb in dl:
            xb = xb.to(device)    # (B,1,L,S,S)
            yb = yb.to(device)    # (B,L)
            opt.zero_grad()
            feats = backbone(xb)
            pred = head(feats)
            loss = criterion(pred, yb)
            loss.backward()
            opt.step()
            running += loss.item() * yb.size(0)
            n += yb.size(0)
        print(f"[Pretraining] epoch {ep:03d}/{cfg.epochs} | MSE {running / max(1,n):.6f}")

    # נשמור רק את ה-Backbone
    state = {"state_dict": backbone.state_dict(), "bands": L, "window": cfg.window}
    torch.save(state, cfg.out_path)
    print(f"[Pretraining] saved → {cfg.out_path}")
    return cfg.out_path, L


# ====================== Fine-Tuning Data & Metrics ======================

class IndianPinesCubes(Dataset):
    """
    דאטהסט לפיקסלים מתוייגים של Indian Pines:
    קולט X:(H,W,L) ו-y:(H,W) עם לייבלים 0..C (0=לא מאומן).
    מוציא חלון (1,L,S,S) ותווית ב-1..C.
    """
    def __init__(self, X: np.ndarray, Y: np.ndarray, window: int = 5, ignore_label: int = 0):
        assert window % 2 == 1
        self.pad = window // 2
        self.window = window
        self.ignore = ignore_label
        Xz = zscore_per_band_hw_last(X)
        self.Xp = reflect_pad_hw_last(Xz, self.pad)
        self.Y = Y.astype(np.int64)
        ii, jj = np.where(self.Y != self.ignore)
        self.pos = [(i + self.pad, j + self.pad) for i, j in zip(ii, jj)]

    def __len__(self): return len(self.pos)

    def __getitem__(self, idx):
        pi, pj = self.pos[idx]
        p = self.pad; s = self.window
        patch = self.Xp[pi - p: pi + p + 1, pj - p: pj + p + 1, :]  # (S,S,L)
        patch = np.transpose(patch, (2, 0, 1))[None, ...].astype(np.float32)  # (1,L,S,S)
        y = int(self.Y[pi - p, pj - p])
        return torch.from_numpy(patch), torch.tensor(y)


def load_indian_pines(data_dir="data"):
    X = sio.loadmat(os.path.join(data_dir, "Indian_pines_corrected.mat"))["indian_pines_corrected"]
    y = sio.loadmat(os.path.join(data_dir, "Indian_pines_gt.mat"))["indian_pines_gt"]
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    return X, y


def split_mask_per_pixel(labels, train_ratio=0.5, ignore_label=0, seed=7):
    rng = np.random.RandomState(seed)
    mask = (labels != ignore_label)
    idx = np.transpose(np.nonzero(mask))
    rng.shuffle(idx)
    n_train = int(len(idx) * train_ratio)
    train_idx, test_idx = idx[:n_train], idx[n_train:]
    train_mask = np.zeros_like(labels, dtype=bool)
    test_mask = np.zeros_like(labels, dtype=bool)
    for i, j in train_idx: train_mask[i, j] = True
    for i, j in test_idx:  test_mask[i, j] = True
    return train_mask, test_mask


def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if t == 0:  # ignore label
            continue
        cm[t-1, p-1] += 1
    return cm


def hsi_scores_from_cm(cm):
    eps = 1e-12
    per_class_acc = np.diag(cm) / (cm.sum(axis=1) + eps)
    AA = per_class_acc.mean()
    OA = np.diag(cm).sum() / (cm.sum() + eps)
    rows = cm.sum(axis=1)
    cols = cm.sum(axis=0)
    pe = (rows * cols).sum() / (cm.sum()**2 + eps)
    kappa = (OA - pe) / (1 - pe + eps)
    return OA, AA, kappa, per_class_acc


# ====================== Classifier Model (same head as אצלך) ======================

class HSI3DCNN_IndianPines(nn.Module):
    def __init__(self, in_bands=200, num_classes=16, fc_dim=128):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 2, kernel_size=(7,3,3))
        self.conv2 = nn.Conv3d(2, 8, kernel_size=(3,3,3))
        self.fc1 = nn.Linear(8*(in_bands-8), fc_dim)  # גנרי לפי מספר ה-bands
        self.fc2 = nn.Linear(fc_dim, num_classes)

        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='relu'); nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight); nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# ====================== Train/Eval Loops ======================

def train_one_epoch_cls(model, loader, optimizer, criterion, device):
    model.train()
    total, correct, running = 0, 0, 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb-1)
        loss.backward()
        optimizer.step()
        running += loss.item() * yb.size(0)
        pred = logits.argmax(1) + 1
        correct += (pred == yb).sum().item()
        total += yb.size(0)
    return running/total, correct/total


@torch.no_grad()
def evaluate_cls(model, loader, criterion, num_classes, device):
    model.eval()
    total, correct, running = 0, 0, 0.0
    yts, yps = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb-1)
        running += loss.item() * yb.size(0)
        pred = logits.argmax(1) + 1
        correct += (pred == yb).sum().item()
        total += yb.size(0)
        yts.append(yb.detach().cpu().numpy())
        yps.append(pred.detach().cpu().numpy())
    y_true = np.concatenate(yts)
    y_pred = np.concatenate(yps)
    cm = confusion_matrix(y_true, y_pred, num_classes)
    OA, AA, kappa, _ = hsi_scores_from_cm(cm)
    return running/total, correct/total, OA, AA, kappa, cm


# ====================== Plotting ======================

def plot_curves(csv_path, out_dir):
    epochs, trL, trA, teL, teA, OA, AA, K = [], [], [], [], [], [], [], []
    with open(csv_path, "r") as f:
        rd = csv.DictReader(f)
        for row in rd:
            epochs.append(int(row["epoch"]))
            trL.append(float(row["train_loss"]))
            trA.append(float(row["train_acc"]))
            teL.append(float(row["test_loss"]))
            teA.append(float(row["test_acc"]))
            OA.append(float(row["OA"]))
            AA.append(float(row["AA"]))
            K.append(float(row["kappa"]))

    # Loss/Accuracy
    plt.figure()
    plt.plot(epochs, trL, label="train loss")
    plt.plot(epochs, teL, label="test loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "loss.png")); plt.close()

    plt.figure()
    plt.plot(epochs, trA, label="train acc")
    plt.plot(epochs, teA, label="test acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "accuracy.png")); plt.close()

    # OA / AA / Kappa
    plt.figure()
    plt.plot(epochs, OA, label="OA")
    plt.plot(epochs, AA, label="AA")
    plt.plot(epochs, K,  label="Kappa")
    plt.xlabel("epoch"); plt.ylabel("score"); plt.legend(); plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "oa_aa_kappa.png")); plt.close()


# ====================== Main: All-in-one ======================

@dataclass
class AllInOneConfig:
    # Pretraining
    pt_root: str = "generated_hs"
    window: int = 5
    mask_ratio: float = 0.15
    pre_epochs: int = 40
    pre_bs: int = 256
    pre_lr: float = 1e-3
    # Fine-tuning
    data_dir: str = "data"  # מכיל את Indian_pines_corrected.mat ו-Indian_pines_gt.mat
    classes: int = 16
    ft_epochs: int = 120
    ft_bs: int = 256
    ft_lr: float = 5e-4
    train_ratio: float = 0.5
    # Misc
    seed: int = 42
    out_dir: str = "runs_all_in_one"


def run_all(cfg: AllInOneConfig):
    setup_seed(cfg.seed)
    device = get_device()
    os.makedirs(cfg.out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(cfg.out_dir, f"run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)

    # ---------- (1) PRETRAIN ----------
    print("\n=== [1/2] Pretraining on generated_hs/*.pt ===")
    pre_cfg = PretrainConfig(
        pt_root=cfg.pt_root, window=cfg.window, mask_ratio=cfg.mask_ratio,
        batch_size=cfg.pre_bs, epochs=cfg.pre_epochs, lr=cfg.pre_lr,
        seed=cfg.seed, out_path=os.path.join(run_dir, "pretrained_backbone_from_pt.pth")
    )
    pretrain_path, L_pre = pretrain_self_supervised(pre_cfg)

    # ---------- (2) FINE-TUNE on Indian Pines ----------
    print("\n=== [2/2] Fine-Tuning on Indian Pines ===")
    X = sio.loadmat(os.path.join(cfg.data_dir, "Indian_pines_corrected.mat"))["indian_pines_corrected"].astype(np.float32)
    y = sio.loadmat(os.path.join(cfg.data_dir, "Indian_pines_gt.mat"))["indian_pines_gt"].astype(np.int64)

    train_mask, test_mask = split_mask_per_pixel(y, train_ratio=cfg.train_ratio, ignore_label=0, seed=cfg.seed)
    tr_ds = IndianPinesCubes(X, np.where(train_mask, y, 0), window=cfg.window)
    te_ds = IndianPinesCubes(X, np.where(test_mask,  y, 0), window=cfg.window)

    tr_dl = DataLoader(tr_ds, batch_size=cfg.ft_bs, shuffle=True,
                       num_workers=2 if device.type=="cuda" else 0,
                       pin_memory=(device.type=="cuda"))
    te_dl = DataLoader(te_ds, batch_size=cfg.ft_bs, shuffle=False,
                       num_workers=2 if device.type=="cuda" else 0,
                       pin_memory=(device.type=="cuda"))

    in_bands_real = X.shape[-1]  # אמור להיות 200 ב-Indian Pines
    model = HSI3DCNN_IndianPines(in_bands=in_bands_real, num_classes=cfg.classes).to(device)

    # טען את ה-Backbone (conv1/conv2) שנלמד ב-pretraining
    sd = torch.load(pretrain_path, map_location="cpu")["state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[Fine-Tuning] loaded backbone weights (missing={len(missing)}, unexpected={len(unexpected)})")

    opt = torch.optim.SGD(model.parameters(), lr=cfg.ft_lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.ft_epochs)
    criterion = nn.CrossEntropyLoss()

    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch","train_loss","train_acc","test_loss","test_acc","OA","AA","kappa"])

    best_OA = 0.0
    for ep in range(1, cfg.ft_epochs+1):
        tr_loss, tr_acc = train_one_epoch_cls(model, tr_dl, opt, criterion, device)
        te_loss, te_acc, OA, AA, kappa, cm = evaluate_cls(model, te_dl, criterion, cfg.classes, device)

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, tr_loss, tr_acc, te_loss, te_acc, OA, AA, kappa])

        if OA > best_OA:
            best_OA = OA
            torch.save({"state_dict": model.state_dict()}, os.path.join(run_dir, "best_finetuned.pth"))

        sch.step()
        print(f"[Fine-Tuning] ep {ep:03d}/{cfg.ft_epochs} | "
              f"train L {tr_loss:.4f} A {tr_acc:.4f} | "
              f"test L {te_loss:.4f} A {te_acc:.4f} | OA {OA:.4f} AA {AA:.4f} κ {kappa:.4f}")

    # שרטוט גרפים
    plot_curves(csv_path, run_dir)
    print(f"\n[Done] Best OA={best_OA:.4f}")
    print(f"Artifacts: {run_dir}/pretrained_backbone_from_pt.pth , {run_dir}/best_finetuned.pth , {csv_path}")
    print(f"Plots: {run_dir}/loss.png , {run_dir}/accuracy.png , {run_dir}/oa_aa_kappa.png")


if __name__ == "__main__":
    cfg = AllInOneConfig(
        pt_root="generated_hs",   # כאן לשים את 600 קבצי ה-.pt בפורמט [L,H,W]
        window=5,
        mask_ratio=0.15,
        pre_epochs=40,
        pre_bs=256,
        pre_lr=1e-3,
        data_dir="data",          # לשים שם את קבצי ה-mat של Indian Pines
        classes=16,
        ft_epochs=120,
        ft_bs=256,
        ft_lr=5e-4,
        train_ratio=0.5,
        seed=42,
        out_dir="runs_all_in_one"
    )
    run_all(cfg)
