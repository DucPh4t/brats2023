import numpy as np
from medpy.metric.binary import hd95

def calc_dice(pred, target):
    smooth = 1e-5
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def calc_dice_3d(pred_vol: np.ndarray, target_vol: np.ndarray):
    """[PAPER FIX] 3D Dice evaluation: smooth=0. Xử lý empty case: both empty->1.0, one empty->0.0."""
    intersection = (pred_vol * target_vol).sum()
    denom = pred_vol.sum() + target_vol.sum()
    if denom == 0:
        return 1.0  # Both empty = perfect match
    return float(2. * intersection / denom)

def calc_hd95_3d(pred_vol: np.ndarray, target_vol: np.ndarray, voxelspacing=None):
    """[PAPER FIX] Tính HD95. Empty-case fallback dùng đường chéo volume theo spacing nếu có."""
    if pred_vol.sum() == 0 and target_vol.sum() == 0:
        return 0.0
    if pred_vol.sum() == 0 or target_vol.sum() == 0:
        shape = np.asarray(pred_vol.shape, dtype=np.float32)
        spacing = np.ones(3, dtype=np.float32) if voxelspacing is None else np.asarray(voxelspacing, dtype=np.float32)
        return float(np.linalg.norm(shape * spacing))
    return hd95(pred_vol, target_vol, voxelspacing=voxelspacing)
