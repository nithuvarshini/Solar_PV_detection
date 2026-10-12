import cv2
import numpy as np
import torch
import torch.nn.functional as F
from src.utils import setup_logger

logger = setup_logger("explainability")

class GradCAM:
    """
    Grad-CAM: Gradient-weighted Class Activation Mapping for PyTorch CNNs.
    Highlights defect-discriminative regions in Solar PV cell Electroluminescence images.
    """
    def __init__(self, model, target_layer_name="base_model.layer4"):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_layer = self._get_target_layer(target_layer_name)
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _get_target_layer(self, layer_name):
        # Traverse model module tree
        modules = layer_name.split(".")
        curr = self.model
        for m in modules:
            curr = getattr(curr, m)
        return curr

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate_heatmap(self, input_tensor, target_class=None, device="cpu"):
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True

        logits = self.model(input_tensor)

        if target_class is None:
            target_class = torch.argmax(logits, dim=1).item()

        self.model.zero_grad()
        target_score = logits[0, target_class]
        target_score.backward()

        # Compute channel-wise weights via Global Average Pooling of gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.size(0)):
            activations[i] *= pooled_gradients[i]

        # Class activation map
        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0) # ReLU
        
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap) # Normalize to [0, 1]

        return heatmap, target_class

def overlay_gradcam(image_np, heatmap, alpha=0.5):
    """
    Overlay Grad-CAM heatmap on original image using OpenCV JET colormap.
    """
    if len(image_np.shape) == 2:
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    else:
        image_rgb = image_np.copy()
        
    h, w, _ = image_rgb.shape
    resized_heatmap = cv2.resize(heatmap, (w, h))
    
    # Convert heatmap to 8-bit unsigned integer
    heatmap_uint8 = np.uint8(255 * resized_heatmap)
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap with original image
    overlaid = cv2.addWeighted(image_rgb, 1 - alpha, color_heatmap, alpha, 0)
    return overlaid, resized_heatmap

def compute_shap_feature_importance(xgb_model, feature_names):
    """
    Compute tree feature importance score breakdown (SHAP proxy feature attribution).
    """
    importances = xgb_model.feature_importances_
    importance_dict = dict(zip(feature_names, importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_importance
