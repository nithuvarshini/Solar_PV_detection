import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFilter
from src.utils import set_seed, setup_logger

logger = setup_logger("data_loader")

# ──────────────────────────────────────────────────────────────────────────────
# Universal Dataset class (works for both synthetic and ELPV real images)
# ──────────────────────────────────────────────────────────────────────────────

class SolarPVDataset(Dataset):
    """
    PyTorch Dataset for Solar PV EL Cell Images & Tabular Telemetries.
    Works with both synthetic generated images and real ELPV/Kaggle datasets.
    """
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        # Telemetry columns – filled with computed image stats if unavailable
        self._telemetry_cols = [
            'Voc', 'Isc', 'Fill_Factor', 'Series_Resistance',
            'Shunt_Resistance', 'Temperature_Delta', 'EL_Intensity_Mean', 'EL_Intensity_Std'
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_name'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])

        if self.transform:
            image = self.transform(image)

        # Build telemetry vector; use defaults if columns are missing
        tel_vals = []
        for col in self._telemetry_cols:
            tel_vals.append(float(row.get(col, 0.0)) if col in row.index else 0.0)
        tabular_features = torch.tensor(tel_vals, dtype=torch.float32)

        return image, tabular_features, label

# ──────────────────────────────────────────────────────────────────────────────
# Synthetic dataset generator (kept as fallback when no real data is available)
# ──────────────────────────────────────────────────────────────────────────────

def generate_synthetic_solar_cell(class_id, size=64):
    """
    Generate a synthetic Solar Cell EL image with per-class defect patterns.
    Class 0: Healthy  |  Class 1: Micro-Crack  |  Class 2: Finger Interrupt  |  Class 3: Delamination
    """
    base_intensity = np.random.randint(145, 195)
    img = np.ones((size, size), dtype=np.uint8) * base_intensity
    noise = np.random.normal(0, 10, (size, size)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    bb1, bb2 = int(size * 0.33), int(size * 0.67)
    draw.line([(bb1, 0), (bb1, size)], fill=35, width=3)
    draw.line([(bb2, 0), (bb2, size)], fill=35, width=3)

    for y in range(8, size, max(6, size // 20)):
        if class_id == 2 and np.random.rand() < 0.45:
            draw.line([(0, y), (int(size * 0.38), y)], fill=65, width=1)
            draw.line([(int(size * 0.62), y), (size, y)], fill=65, width=1)
        else:
            draw.line([(0, y), (size, y)], fill=65, width=1)

    img_arr = np.array(pil_img)

    if class_id == 1:
        x1 = np.random.randint(5, size // 4)
        y1 = np.random.randint(5, size // 4)
        x2 = np.random.randint(size * 3 // 4, size - 5)
        y2 = np.random.randint(size * 3 // 4, size - 5)
        cv2.line(img_arr, (x1, y1), (x2, y2), (18,), max(1, size // 64))
        cv2.line(img_arr, (x1 + 5, y1), (x2 - 5, y2 + 3), (30,), 1)

    elif class_id == 3:
        cx = np.random.randint(size // 5, size * 4 // 5)
        cy = np.random.randint(size // 5, size * 4 // 5)
        r = max(5, size // 8)
        cv2.circle(img_arr, (cx, cy), r, (25,), -1)
        cv2.ellipse(img_arr, (cx + r // 2, cy + r // 4),
                    (r + 5, r // 2), 30, 0, 360, (40,), -1)
        img_arr = cv2.GaussianBlur(img_arr, (5, 5), 0)

    img_arr[0:2, :] = 18
    img_arr[-2:, :] = 18
    img_arr[:, 0:2] = 18
    img_arr[:, -2:] = 18

    return Image.fromarray(img_arr).convert("RGB")


def generate_synthetic_tabular_telemetry(class_id):
    """Generate physically correlated solar panel telemetry for each defect class."""
    params = {
        0: dict(voc=(0.62, 0.65), isc=(9.0, 9.5), ff=(78.0, 82.0),
                rs=(0.010, 0.020), rsh=(400, 500), td=(0.5, 2.0),
                el_m=(160, 180), el_s=(15, 25)),
        1: dict(voc=(0.57, 0.61), isc=(8.7, 9.2), ff=(71.0, 76.0),
                rs=(0.025, 0.040), rsh=(250, 350), td=(2.5, 5.0),
                el_m=(140, 160), el_s=(30, 45)),
        2: dict(voc=(0.59, 0.63), isc=(8.0, 8.6), ff=(73.0, 77.0),
                rs=(0.035, 0.055), rsh=(300, 420), td=(3.0, 6.0),
                el_m=(145, 165), el_s=(25, 38)),
        3: dict(voc=(0.51, 0.56), isc=(7.5, 8.3), ff=(62.0, 70.0),
                rs=(0.050, 0.080), rsh=(100, 220), td=(6.5, 12.0),
                el_m=(115, 140), el_s=(45, 65)),
    }
    p = params[class_id]
    return {
        'Voc': round(np.random.uniform(*p['voc']), 4),
        'Isc': round(np.random.uniform(*p['isc']), 4),
        'Fill_Factor': round(np.random.uniform(*p['ff']), 2),
        'Series_Resistance': round(np.random.uniform(*p['rs']), 4),
        'Shunt_Resistance': round(np.random.uniform(*p['rsh']), 2),
        'Temperature_Delta': round(np.random.uniform(*p['td']), 2),
        'EL_Intensity_Mean': round(np.random.uniform(*p['el_m']), 2),
        'EL_Intensity_Std': round(np.random.uniform(*p['el_s']), 2),
    }


def load_or_generate_synthetic(config):
    """Create synthetic EL dataset (used when dataset_mode == 'synthetic')."""
    set_seed(config['project']['seed'])
    raw_dir = config['paths']['data_raw']
    img_dir = os.path.join(raw_dir, "pv_images")
    os.makedirs(img_dir, exist_ok=True)
    csv_path = os.path.join(raw_dir, "solar_pv_dataset.csv")

    if os.path.exists(csv_path):
        logger.info(f"Loading existing dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        return df, img_dir

    logger.info(f"Generating synthetic dataset ({config['dataset']['num_samples']} samples)…")
    data = []
    n = config['dataset']['num_samples']
    num_cls = config['dataset']['num_classes']
    spc = n // num_cls

    for c in range(num_cls):
        for i in range(spc):
            idx = c * spc + i
            img_name = f"pv_cell_{idx:04d}_cls{c}.png"
            img_path = os.path.join(img_dir, img_name)
            pil = generate_synthetic_solar_cell(class_id=c, size=config['dataset']['image_size'])
            pil.save(img_path)
            tel = generate_synthetic_tabular_telemetry(class_id=c)
            rec = {'sample_id': idx, 'image_name': img_name, 'label': c,
                   'defect_class': config['dataset']['class_names'][c]}
            rec.update(tel)
            data.append(rec)

    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    logger.info(f"Synthetic dataset saved -> {csv_path} ({len(df)} samples)")
    return df, img_dir


# ──────────────────────────────────────────────────────────────────────────────
# Real ELPV dataset loader
# ──────────────────────────────────────────────────────────────────────────────

def load_elpv_dataset(config):
    """
    Load the real ELPV dataset organised by prepare_elpv.py.

    Expected folder structure after prepare_elpv.py:
        data/elpv/
            images/          ← all .png files
            elpv_dataset.csv ← columns: image_name, label, defect_class,
                                         Voc, Isc, Fill_Factor, Series_Resistance,
                                         Shunt_Resistance, Temperature_Delta,
                                         EL_Intensity_Mean, EL_Intensity_Std
    """
    real_dir = config['dataset']['real_data_dir']
    csv_path = os.path.join(real_dir, "elpv_dataset.csv")
    img_dir  = os.path.join(real_dir, "images")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"ELPV dataset CSV not found at {csv_path}.\n"
            "Please run:  python prepare_elpv.py  first to download and prepare the dataset."
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded ELPV dataset: {len(df)} samples from {csv_path}")
    return df, img_dir


# ──────────────────────────────────────────────────────────────────────────────
# Master dataset loader (dispatches between synthetic and real)
# ──────────────────────────────────────────────────────────────────────────────

def load_dataset(config):
    """Dispatch to the correct dataset based on config['dataset']['dataset_mode']."""
    mode = config['dataset'].get('dataset_mode', 'synthetic')
    if mode == 'elpv':
        return load_elpv_dataset(config)
    return load_or_generate_synthetic(config)


def get_dataloaders(config):
    """
    Prepare PyTorch DataLoaders with augmentation for train / val / test splits.
    Automatically detects synthetic or real ELPV mode from config.
    """
    df, img_dir = load_dataset(config)

    df = df.sample(frac=1.0, random_state=config['project']['seed']).reset_index(drop=True)

    n = len(df)
    train_end = int(n * config['dataset']['split_ratios']['train'])
    val_end   = train_end + int(n * config['dataset']['split_ratios']['val'])

    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df   = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df  = df.iloc[val_end:].reset_index(drop=True)

    img_size = config['dataset']['image_size']
    mean     = config['preprocessing']['mean']
    std      = config['preprocessing']['std']

    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_ds = SolarPVDataset(train_df, img_dir, transform=train_transform)
    val_ds   = SolarPVDataset(val_df,   img_dir, transform=val_test_transform)
    test_ds  = SolarPVDataset(test_df,  img_dir, transform=val_test_transform)

    bs = config['model']['batch_size']
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0)

    logger.info(f"DataLoaders ready │ train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")
    return train_loader, val_loader, test_loader, (train_df, val_df, test_df)
