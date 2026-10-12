import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from src.utils import setup_logger

logger = setup_logger("preprocessing")

def apply_clahe(image_np, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Apply CLAHE contrast enhancement to solar cell images."""
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()
        
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced = clahe.apply(gray)
    return enhanced

def extract_texture_features(image_np):
    """
    Extract advanced hand-crafted texture & edge features from EL images.
    Features: Intensity mean, std, edge density (Sobel), GLCM approximations.
    """
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()
        
    # Contrast & Intensity stats
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    skewness = np.mean(((gray - mean_val) / (std_val + 1e-6)) ** 3)
    
    # Sobel Edge Gradient Magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobelx**2 + sobely**2)
    edge_density = np.mean(grad_mag > 50)
    grad_std = np.std(grad_mag)
    
    # Simple GLCM proxy features: Spatial co-occurrence differences
    diff_h = np.abs(gray[:, :-1].astype(float) - gray[:, 1:].astype(float))
    diff_v = np.abs(gray[:-1, :].astype(float) - gray[1:, :].astype(float))
    glcm_contrast = np.mean(diff_h**2) + np.mean(diff_v**2)
    glcm_homogeneity = np.mean(1.0 / (1.0 + diff_h)) + np.mean(1.0 / (1.0 + diff_v))
    glcm_energy = np.mean(gray**2) / (255.0**2)
    
    return {
        'img_mean': mean_val,
        'img_std': std_val,
        'img_skew': skewness,
        'edge_density': edge_density,
        'grad_std': grad_std,
        'glcm_contrast': glcm_contrast,
        'glcm_homogeneity': glcm_homogeneity,
        'glcm_energy': glcm_energy
    }

def prepare_tabular_and_texture_dataset(df, img_dir):
    """Extract features from images and combine with tabular solar panel telemetries."""
    import os
    from PIL import Image
    
    feature_list = []
    for idx, row in df.iterrows():
        img_path = os.path.join(img_dir, row['image_name'])
        img = np.array(Image.open(img_path))
        txt_feats = extract_texture_features(img)
        
        combined = {
            'Voc': row['Voc'],
            'Isc': row['Isc'],
            'Fill_Factor': row['Fill_Factor'],
            'Series_Resistance': row['Series_Resistance'],
            'Shunt_Resistance': row['Shunt_Resistance'],
            'Temperature_Delta': row['Temperature_Delta'],
            'EL_Intensity_Mean': row['EL_Intensity_Mean'],
            'EL_Intensity_Std': row['EL_Intensity_Std']
        }
        combined.update(txt_feats)
        feature_list.append(combined)
        
    feat_df = pd.DataFrame(feature_list)
    return feat_df

def compute_dimensionality_reduction(X, n_pca=2, tsne_perplexity=30, random_state=42):
    """Compute PCA and t-SNE 2D projections for feature visualization."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA 2D
    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_
    
    # t-SNE 2D
    perp = min(tsne_perplexity, X.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state)
    X_tsne = tsne.fit_transform(X_scaled)
    
    return X_pca, X_tsne, explained_variance, scaler
