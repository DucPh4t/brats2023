import numpy as np
from .normalization import (
    min_max,
    zscore_clip,
    zscore_clip_clahe,
    zscore_clip_custom,
    zscore_raw,
)

def get_preprocessor(norm_type, config=None):
    """Factory function to return the correct preprocessing function."""
    config = config or {}
    if norm_type in ("min_max", "minmax"):
        return lambda vol, modality=None: min_max(vol)
    elif norm_type == "zscore_clip":
        return lambda vol, modality=None: zscore_clip(vol)
    elif norm_type == "zscore_clip_custom":
        return lambda vol, modality=None: zscore_clip_custom(
            vol,
            config=config.get("normalization_params", {}),
            modality=modality,
        )
    elif norm_type == "zscore_clip_clahe":
        return lambda vol, modality=None: zscore_clip_clahe(
            vol,
            config=config.get("clahe", {}),
            modality=modality,
        )
    elif norm_type == "zscore_raw":
        return lambda vol, modality=None: zscore_raw(vol)
    else:
        # Default fallback: No normalization
        return lambda vol, modality=None: vol.astype(np.float32)
