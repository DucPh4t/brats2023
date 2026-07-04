import numpy as np

def _brain_mask(vol_3d):
    return vol_3d > 0

def _zscore_masked(vol_3d, brain_mask):
    out = np.zeros_like(vol_3d, dtype=np.float32)
    if brain_mask.sum() == 0:
        return out

    brain_voxels = vol_3d[brain_mask]
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    out[brain_mask] = (brain_voxels - mean) / (std + 1e-8)
    return out

def _clip_to_unit(vol_3d, brain_mask, lower=0.5, upper=99.5):
    if brain_mask.sum() == 0:
        return np.zeros_like(vol_3d, dtype=np.float32)

    p_min = np.percentile(vol_3d[brain_mask], lower)
    p_max = np.percentile(vol_3d[brain_mask], upper)
    clipped = np.clip(vol_3d, p_min, p_max).astype(np.float32)
    clipped[~brain_mask] = 0

    out = np.zeros_like(clipped, dtype=np.float32)
    if p_max > p_min:
        out[brain_mask] = (clipped[brain_mask] - p_min) / (p_max - p_min)
    return out

def _normalize_modalities(modalities):
    if modalities in (None, "all"):
        return "all"
    return {str(mod).lower() for mod in modalities}

def min_max(vol_3d):
    """Brain-masked 3D Volume Min-Max Normalization (Fair Ablation Baseline).
    - Brain mask từ volume > 0
    - Min/Max chỉ trên brain voxels
    - Background giữ nguyên = 0
    """
    out = np.zeros_like(vol_3d, dtype=np.float32)
    brain_mask = vol_3d > 0
    if brain_mask.sum() == 0:
        return out
        
    min_val = vol_3d[brain_mask].min()
    max_val = vol_3d[brain_mask].max()
    
    if max_val > min_val:
        out[brain_mask] = (vol_3d[brain_mask] - min_val) / (max_val - min_val)
    else:
        out[brain_mask] = vol_3d[brain_mask]
    return out

def zscore_clip(vol_3d):
    """[PAPER FIX] Z-Score (Chuẩn A4): 
    - Percentile clip (0.5, 99.5) trước z-score
    - Masking (vol > 0) để chỉ normalize não, background=0
    """
    out = np.zeros_like(vol_3d, dtype=np.float32)
    brain_mask = vol_3d > 0
    if brain_mask.sum() == 0:
        return out
        
    # Percentile clipping (0.5, 99.5) to remove extreme outliers before z-score
    p_min = np.percentile(vol_3d[brain_mask], 0.5)
    p_max = np.percentile(vol_3d[brain_mask], 99.5)
    
    clipped = np.clip(vol_3d, p_min, p_max)   # clip trên 3D volume
    clipped[~brain_mask] = 0                  # restore background
    brain_clipped = clipped[brain_mask]       # lấy brain voxels đã clip
    
    mean = brain_clipped.mean()
    std  = brain_clipped.std()
    
    out[brain_mask] = (brain_clipped - mean) / (std + 1e-8)
    return out

def zscore_clip_custom(vol_3d, config=None, modality=None):
    """Configurable percentile clipping followed by brain-masked z-score.

    This keeps the Exp004 preprocessing family but lets B-revisit experiments
    test whether a wider/tighter percentile window preserves ET cues better.
    """
    config = config or {}
    percentiles = config.get("clip_percentiles", [0.5, 99.5])
    lower, upper = float(percentiles[0]), float(percentiles[1])

    brain_mask = _brain_mask(vol_3d)
    if brain_mask.sum() == 0:
        return np.zeros_like(vol_3d, dtype=np.float32)

    clipped_unit = _clip_to_unit(vol_3d, brain_mask, lower=lower, upper=upper)
    return _zscore_masked(clipped_unit, brain_mask)

def _apply_clahe_slicewise(vol_unit, brain_mask, config):
    try:
        from skimage import exposure
    except ImportError as exc:
        raise ImportError("CLAHE preprocessing requires scikit-image (`skimage`).") from exc

    clip_limit = float(config.get("clip_limit", 0.01))
    kernel_size = config.get("kernel_size", 32)
    nbins = int(config.get("nbins", 256))
    min_brain_pixels = int(config.get("min_brain_pixels", 100))

    out = np.zeros_like(vol_unit, dtype=np.float32)
    for z in range(vol_unit.shape[2]):
        slice_mask = brain_mask[:, :, z]
        if int(slice_mask.sum()) < min_brain_pixels:
            out[:, :, z] = vol_unit[:, :, z]
            continue

        enhanced = exposure.equalize_adapthist(
            vol_unit[:, :, z],
            kernel_size=kernel_size,
            clip_limit=clip_limit,
            nbins=nbins,
        ).astype(np.float32)
        enhanced[~slice_mask] = 0
        out[:, :, z] = enhanced
    return out

def zscore_clip_clahe(vol_3d, config=None, modality=None):
    """Percentile clip -> [0,1] -> optional per-slice CLAHE -> z-score.

    CLAHE is applied only to configured modalities, so B-revisit can test
    T1ce-only or FLAIR+T1ce without changing the rest of the pipeline.
    """
    config = config or {}
    modalities = _normalize_modalities(config.get("modalities", "all"))
    mod = str(modality).lower() if modality is not None else None
    apply_clahe = modalities == "all" or mod in modalities

    percentiles = config.get("clip_percentiles", [0.5, 99.5])
    lower, upper = float(percentiles[0]), float(percentiles[1])

    brain_mask = _brain_mask(vol_3d)
    if brain_mask.sum() == 0:
        return np.zeros_like(vol_3d, dtype=np.float32)

    clipped_unit = _clip_to_unit(vol_3d, brain_mask, lower=lower, upper=upper)
    if apply_clahe:
        clipped_unit = _apply_clahe_slicewise(clipped_unit, brain_mask, config)
    return _zscore_masked(clipped_unit, brain_mask)

def zscore_raw(vol_3d):
    """Z-score cơ bản KHÔNG có percentile clipping."""
    out = np.zeros_like(vol_3d, dtype=np.float32)
    brain_mask = vol_3d > 0
    if brain_mask.sum() == 0:
        return out
        
    brain_voxels = vol_3d[brain_mask]
    mean = brain_voxels.mean()
    std  = brain_voxels.std()
    
    out[brain_mask] = (brain_voxels - mean) / (std + 1e-8)
    return out
