import os
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
import numpy as np

class LogWriter:
    def __init__(self, file_path, train=True):
        os.makedirs(file_path, exist_ok=True)
        self.file_path = os.path.join(file_path, "train.txt" if train else "test.txt")

    def write(self, text):
        print(text)
        with open(self.file_path, 'a') as file:
            file.write(text + '\n')


def sam_loss_fn(pred, target, eps=1e-8):
    """Differentiable SAM loss (cosine distance). pred, target: [1, C, H, W]."""
    pred_flat = pred.squeeze(0).permute(1, 2, 0).reshape(-1, pred.shape[1])
    target_flat = target.squeeze(0).permute(1, 2, 0).reshape(-1, target.shape[1])
    cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=1, eps=eps)
    return (1 - cos_sim).mean()

def loss_fn(pred, target, loss_type='L2', lambda_value=0.7):
    target = target.detach()
    pred = pred.float()
    target  = target.float()
    if loss_type == 'L2':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'L1':
        loss = F.l1_loss(pred, target)
    elif loss_type == 'SSIM':
        loss = 1 - ssim(pred, target, data_range=1, size_average=True)
    elif loss_type == 'Fusion1':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion2':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion3':
        loss = lambda_value * F.mse_loss(pred, target) + (1-lambda_value) * F.l1_loss(pred, target)
    elif loss_type == 'Fusion4':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value) * (1 - ms_ssim(pred, target, data_range=1, size_average=True))
    elif loss_type == 'Fusion_hinerv':
        loss = lambda_value * F.l1_loss(pred, target) + (1-lambda_value)  * (1 - ms_ssim(pred, target, data_range=1, size_average=True, win_size=5))
    else:
        raise ValueError(f"Unknown loss_type: {loss_type!r}")
    return loss

def compute_sam(original, reconstructed, eps=1e-8):
    """
    Compute mean Spectral Angle Mapper (SAM) between original and reconstructed hyperspectral images.
    original, reconstructed: [H, W, C] arrays.
    """
    dot_product = np.sum(original * reconstructed, axis=-1)  # [H, W]
    norm_orig = np.linalg.norm(original, axis=-1)
    norm_recon = np.linalg.norm(reconstructed, axis=-1)
    cos_theta = dot_product / (norm_orig * norm_recon + eps)
    cos_theta = np.clip(cos_theta, -1, 1)  # to avoid NaNs
    sam = np.arccos(cos_theta)  # [H, W]
    mean_sam = np.mean(sam)
    return mean_sam
