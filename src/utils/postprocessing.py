import numpy as np
from scipy import ndimage


REGION_NAMES = ("WT", "TC", "ET")


def _remove_small_components(mask: np.ndarray, min_voxels: int = 0, keep_largest: bool = False) -> np.ndarray:
    if min_voxels <= 0 and not keep_largest:
        return mask.astype(np.float32)

    labeled, num_components = ndimage.label(mask > 0)
    if num_components == 0:
        return mask.astype(np.float32)

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0

    if keep_largest:
        keep_labels = {int(component_sizes.argmax())}
    else:
        keep_labels = set(np.where(component_sizes >= min_voxels)[0].astype(int).tolist())
        keep_labels.discard(0)

    if not keep_labels:
        return np.zeros_like(mask, dtype=np.float32)

    return np.isin(labeled, list(keep_labels)).astype(np.float32)


def enforce_region_hierarchy(pred_volume: np.ndarray) -> np.ndarray:
    """Ensure BraTS nested regions stay valid: ET subset TC subset WT."""
    fixed = pred_volume.astype(np.float32).copy()
    fixed[:, 1] = np.logical_or(fixed[:, 1] > 0, fixed[:, 2] > 0)
    fixed[:, 0] = np.logical_or(fixed[:, 0] > 0, fixed[:, 1] > 0)
    return fixed.astype(np.float32)


def postprocess_regions(pred_volume: np.ndarray, config: dict | None = None) -> np.ndarray:
    config = config or {}
    processed = pred_volume.astype(np.float32).copy()
    min_voxels = config.get("min_component_voxels", {})
    keep_largest = config.get("keep_largest", {})

    for channel, region in enumerate(REGION_NAMES):
        processed[:, channel] = _remove_small_components(
            processed[:, channel],
            min_voxels=int(min_voxels.get(region, 0)),
            keep_largest=bool(keep_largest.get(region, False)),
        )

    if config.get("enforce_hierarchy", True):
        processed = enforce_region_hierarchy(processed)

    return processed.astype(np.float32)
