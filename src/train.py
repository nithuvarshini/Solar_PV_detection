import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import numpy as np
import pandas as pd
from src.utils import setup_logger, get_device, set_seed, ensure_directories
from src.data_loader import get_dataloaders
from src.preprocessing import prepare_tabular_and_texture_dataset
from src.models import SolarResNetClassifier, SolarCustomCNN, train_xgboost_classifier

logger = setup_logger("train")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train PyTorch model for 1 epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, _, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data).item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    """Validate PyTorch model performance."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, _, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc

def train_cnn_pipeline(config, train_loader, val_loader, device):
    """Train PyTorch ResNet fine-tuning CNN pipeline."""
    set_seed(config['project']['seed'])
    num_classes = config['dataset']['num_classes']
    backbone = config['model']['cnn_backbone']
    lr = config['model']['learning_rate']
    weight_decay = config['model']['weight_decay']
    epochs = config['model']['num_epochs']
    dropout = config['model']['dropout']

    model = SolarResNetClassifier(num_classes=num_classes, backbone=backbone, pretrained=True, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    models_dir = config['paths']['models_dir']
    best_model_path = os.path.join(models_dir, "best_pv_cnn.pt")

    logger.info(f"Starting CNN Training ({backbone}) for {epochs} epochs...")
    start_time = time.time()

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        logger.info(f"Epoch [{epoch+1:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

    elapsed = time.time() - start_time
    logger.info(f"CNN Training completed in {elapsed:.2f}s. Best Val Acc: {best_val_acc:.4f}")
    
    # Save training history
    with open(os.path.join(models_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

    # Load best weights
    model.load_state_dict(torch.load(best_model_path))
    return model, history

def run_optuna_hpo(config, train_loader, val_loader, device):
    """Run Optuna Automated Hyperparameter Optimization (fast CPU-friendly version)."""
    logger.info("Starting Optuna Hyperparameter Optimization study...")

    n_trials   = config['hpo'].get('n_trials', 3)
    hpo_epochs = config['hpo'].get('hpo_epochs', 1)       # Fast: 1 epoch per trial
    subset_frac = config['hpo'].get('hpo_subset_frac', 0.4)  # Use subset of data

    # Build a lightweight subset DataLoader for HPO
    from torch.utils.data import Subset
    full_ds = train_loader.dataset
    subset_size = max(1, int(len(full_ds) * subset_frac))
    subset_idx = list(range(subset_size))
    hpo_ds = Subset(full_ds, subset_idx)
    hpo_loader = torch.utils.data.DataLoader(
        hpo_ds, batch_size=config['model']['batch_size'], shuffle=True, num_workers=0
    )

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)

        model = SolarResNetClassifier(
            num_classes=config['dataset']['num_classes'],
            backbone=config['model']['cnn_backbone'],
            pretrained=False,          # No download during HPO = faster
            dropout=dropout
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        for _ in range(hpo_epochs):
            train_one_epoch(model, hpo_loader, criterion, optimizer, device)

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        return val_acc

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, timeout=config['hpo'].get('timeout', 300))

    logger.info(f"Optuna HPO Best Parameters: {study.best_params} (Best Acc: {study.best_value:.4f})")
    return study


def execute_full_training_pipeline(config):
    """Master training function running CNN fine-tuning, Optuna HPO, and XGBoost training."""
    ensure_directories(config)
    device = get_device(config['model']['device'])
    logger.info(f"Execution Device: {device}")

    train_loader, val_loader, test_loader, (train_df, val_df, test_df) = get_dataloaders(config)

    # Optuna HPO
    hpo_study = run_optuna_hpo(config, train_loader, val_loader, device)

    # CNN Training
    cnn_model, history = train_cnn_pipeline(config, train_loader, val_loader, device)

    # Preprocessing tabular telemetries + extracted texture features for XGBoost
    img_dir = config["dataset"]["real_data_dir"]
    img_dir = os.path.join(img_dir, "images")
    
    logger.info("Extracting image texture features for XGBoost model...")
    train_feats = prepare_tabular_and_texture_dataset(train_df, img_dir)
    val_feats = prepare_tabular_and_texture_dataset(val_df, img_dir)

    y_train = train_df['label'].values
    y_val = val_df['label'].values

    xgb_model = train_xgboost_classifier(train_feats, y_train, val_feats, y_val)
    xgb_path = os.path.join(config['paths']['models_dir'], "best_pv_xgb.json")
    xgb_model.save_model(xgb_path)
    logger.info(f"XGBoost model saved to {xgb_path}")

    return cnn_model, xgb_model, history, hpo_study, (train_df, val_df, test_df)
