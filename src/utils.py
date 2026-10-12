import os
import random
import logging
import yaml
import numpy as np
import torch

def setup_logger(name="solar_pv_logger", log_file="pipeline.log", level=logging.INFO):
    """Set up a logger with standard formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler(log_file, mode='a')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger

def set_seed(seed=42):
    """Set seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_config(config_path="config/config.yaml"):
    """Load configuration YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def ensure_directories(config):
    """Ensure required directory structures exist."""
    dirs = [
        config['paths']['data_raw'],
        config['paths']['data_processed'],
        config['paths']['models_dir'],
        config['paths']['visualizations_dir'],
        "logs"
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def get_device(requested="auto"):
    """Detect and return PyTorch torch.device."""
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    elif requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")
