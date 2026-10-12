import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import xgboost as xgb
import numpy as np
from src.utils import setup_logger

logger = setup_logger("models")

class SpatialAttention(nn.Module):
    """Spatial Attention Module for defect feature focus on solar cells."""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        att = self.sigmoid(self.conv(combined))
        return x * att

class SolarResNetClassifier(nn.Module):
    """
    Fine-Tuned Transfer Learning CNN (ResNet18 / ResNet50) for Solar PV Defect Detection.
    Includes Spatial Attention and custom classification head.
    """
    def __init__(self, num_classes=4, backbone="resnet18", pretrained=True, dropout=0.3):
        super(SolarResNetClassifier, self).__init__()
        
        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.base_model = models.resnet18(weights=weights)
            in_features = self.base_model.fc.in_features
        elif backbone == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.base_model = models.resnet50(weights=weights)
            in_features = self.base_model.fc.in_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Replace classification layer with identity
        self.base_model.fc = nn.Identity()
        self.attention = SpatialAttention()
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Extract features from convolutional stem (up to layer4)
        x_conv = self.base_model.conv1(x)
        x_conv = self.base_model.bn1(x_conv)
        x_conv = self.base_model.relu(x_conv)
        x_conv = self.base_model.maxpool(x_conv)

        x_conv = self.base_model.layer1(x_conv)
        x_conv = self.base_model.layer2(x_conv)
        x_conv = self.base_model.layer3(x_conv)
        x_conv = self.base_model.layer4(x_conv)

        # Spatial attention filtering
        x_att = self.attention(x_conv)

        # Global average pooling
        x_gap = torch.mean(x_att, dim=[2, 3])
        logits = self.classifier(x_gap)
        return logits

class SolarCustomCNN(nn.Module):
    """Lightweight Custom CNN Baseline Architecture."""
    def __init__(self, num_classes=4, dropout=0.3):
        super(SolarCustomCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 64x64
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)

def train_xgboost_classifier(X_train, y_train, X_val, y_val, params=None):
    """Train XGBoost Gradient Boosting Model on tabular telemetries & texture features."""
    if params is None:
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'mlogloss'
        }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model

class SolarEnsembleModel:
    """Multi-modal Stacking Ensemble combining CNN EL Image Predictions & XGBoost Telemetry Probabilities."""
    def __init__(self, cnn_model, xgb_model, alpha=0.6):
        self.cnn_model = cnn_model
        self.xgb_model = xgb_model
        self.alpha = alpha  # Weight for CNN model (1-alpha for XGBoost)

    def predict_proba(self, images_tensor, tabular_np, device="cpu"):
        self.cnn_model.eval()
        with torch.no_grad():
            images_tensor = images_tensor.to(device)
            cnn_logits = self.cnn_model(images_tensor)
            cnn_probs = F.softmax(cnn_logits, dim=1).cpu().numpy()

        xgb_probs = self.xgb_model.predict_proba(tabular_np)
        
        # Weighted ensemble soft voting
        combined_probs = self.alpha * cnn_probs + (1 - self.alpha) * xgb_probs
        return combined_probs

    def predict(self, images_tensor, tabular_np, device="cpu"):
        probs = self.predict_proba(images_tensor, tabular_np, device)
        return np.argmax(probs, axis=1)
