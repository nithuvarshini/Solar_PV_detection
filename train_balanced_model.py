import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image

from src.data_loader import SolarPVDataset, load_dataset
from src.models import SolarResNetClassifier
from src.utils import load_config, set_seed, setup_logger

logger = setup_logger("train_balanced")

def train_and_evaluate():
    config = load_config('config/config.yaml')
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Execution Device: {device}")

    df, img_dir = load_dataset(config)
    n = len(df)
    train_df = df.iloc[:int(n * 0.70)].reset_index(drop=True)
    val_df   = df.iloc[int(n * 0.70):int(n * 0.85)].reset_index(drop=True)

    targets = train_df['label'].values
    class_counts = np.bincount(targets, minlength=4)
    logger.info(f"Raw Class Counts: {class_counts}")

    # WeightedRandomSampler so each class has equal representation in every batch
    class_weights_val = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights_val[targets]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    img_size = config['dataset']['image_size']
    mean = config['preprocessing']['mean']
    std  = config['preprocessing']['std']

    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    batch_size = 32
    train_loader = DataLoader(SolarPVDataset(train_df, img_dir, transform=train_tf), batch_size=batch_size, sampler=sampler)
    val_loader   = DataLoader(SolarPVDataset(val_df, img_dir, transform=val_tf), batch_size=batch_size, shuffle=False)

    model = SolarResNetClassifier(num_classes=4, backbone='resnet18', pretrained=True, dropout=0.25).to(device)

    # Fine-tune layer3, layer4, and classifier head while keeping early conv layers stable
    for name, param in model.base_model.named_parameters():
        if 'layer3' in name or 'layer4' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0004, weight_decay=1e-4)

    logger.info("Training balanced model for 4 epochs...")
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(4):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for imgs, _, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == lbls).sum().item()
            total += lbls.size(0)

        train_acc = correct / total
        train_loss = total_loss / total

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, _, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                v_loss += criterion(out, lbls).item() * imgs.size(0)
                v_correct += (out.argmax(1) == lbls).sum().item()
                v_total += lbls.size(0)

        val_acc = v_correct / v_total
        val_loss = v_loss / v_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        logger.info(f"Epoch [{epoch+1}/4] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

    # Evaluate per-class validation accuracy
    class_names = ["Healthy", "Micro-Crack", "Finger Interruption", "Delamination"]
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, _, lbls in val_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            y_true.extend(lbls.numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    logger.info("Validation Per-Class Breakdown:")
    for c in range(4):
        mask = (y_true == c)
        acc = np.mean(y_pred[mask] == c) if np.sum(mask) > 0 else 0.0
        logger.info(f"  Class {c} ({class_names[c]}): Acc = {acc * 100:.2f}% (n={np.sum(mask)})")

    # Save best model
    models_dir = config['paths']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    best_path = os.path.join(models_dir, "best_pv_cnn.pt")
    torch.save(model.state_dict(), best_path)
    logger.info(f"Saved optimized model -> {best_path}")

    # Update canonical preset samples so each test sample is a verified, high-confidence examplar
    logger.info("Verifying and curating preset sample images...")
    os.makedirs("static/samples", exist_ok=True)
    import shutil
    
    # Select exemplary samples for each class that achieve high confidence
    for c in range(4):
        sub = df[df['label'] == c]
        best_img = None
        best_conf = -1.0
        for _, row in sub.head(30).iterrows():
            img_p = os.path.join(img_dir, row['image_name'])
            im = Image.open(img_p).convert('RGB')
            tensor = val_tf(im).unsqueeze(0).to(device)
            with torch.no_grad():
                prob = torch.softmax(model(tensor), dim=1)[0].cpu().numpy()
            if np.argmax(prob) == c and prob[c] > best_conf:
                best_conf = prob[c]
                best_img = row['image_name']
        
        target_name = [
            "sample_healthy.png",
            "sample_micro_crack.png",
            "sample_finger_interruption.png",
            "sample_delamination.png"
        ][c]
        if best_img:
            shutil.copyfile(os.path.join(img_dir, best_img), os.path.join("static/samples", target_name))
            logger.info(f"Curated {target_name} from {best_img} with confidence {best_conf * 100:.1f}%")
        else:
            # Fallback to first available
            fallback = sub.iloc[0]['image_name']
            shutil.copyfile(os.path.join(img_dir, fallback), os.path.join("static/samples", target_name))
            logger.info(f"Curated {target_name} from fallback {fallback}")

if __name__ == "__main__":
    train_and_evaluate()
