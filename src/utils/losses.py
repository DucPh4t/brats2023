import torch
import torch.nn.functional as F

def dice_loss(pred, target, smooth=1.):
    """[PAPER FIX] Nhận probabilities (không phải logits). smooth=1.0 để chống nổ gradient."""
    pred = pred.contiguous()
    target = target.contiguous()    
    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def focal_tversky_loss(pred, target, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
    """
    Focal Tversky Loss for imbalanced segmentation.
    - alpha: weight for False Negatives (FN). Set higher (e.g. 0.7) to improve Recall.
    - beta: weight for False Positives (FP).
    - gamma: focal parameter (typical range [1, 3]).
    """
    pred = pred.contiguous()
    target = target.contiguous()

    # TP, FN, FP
    tp = (pred * target).sum(dim=(2, 3))
    fn = ((1 - pred) * target).sum(dim=(2, 3))
    fp = (pred * (1 - target)).sum(dim=(2, 3))

    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    loss = (1 - tversky_index) ** gamma
    
    return loss.mean()

def dice_bce_loss(pred, target, dice_weight=0.5, smooth=1.):
    """
    Combo Loss = dice_weight * DiceLoss + (1 - dice_weight) * BCELoss
    - Dice Loss: giữ vững chất lượng vùng lớn (WT)
    - BCE Loss: phạt ổn định từng pixel
    """
    d_loss = dice_loss(pred, target, smooth=smooth)
    bce_loss = F.binary_cross_entropy(pred, target)
    return dice_weight * d_loss + (1.0 - dice_weight) * bce_loss

def weighted_dice_bce_loss(pred, target, dice_weight=0.5, bce_channel_weights=None, smooth=1.):
    """
    Dice + channel-weighted BCE for BraTS region imbalance.

    Channel order is [WT, TC, ET]. The Dice part stays unchanged while the BCE
    part can mildly emphasize smaller regions such as TC/ET.
    """
    d_loss = dice_loss(pred, target, smooth=smooth)

    bce = F.binary_cross_entropy(pred, target, reduction="none")
    if bce_channel_weights is not None:
        if len(bce_channel_weights) != pred.shape[1]:
            raise ValueError(
                f"bce_channel_weights must have {pred.shape[1]} values, got {len(bce_channel_weights)}"
            )
        weights = torch.tensor(
            bce_channel_weights,
            dtype=pred.dtype,
            device=pred.device,
        ).view(1, pred.shape[1], 1, 1)
        bce = bce * weights

    return dice_weight * d_loss + (1.0 - dice_weight) * bce.mean()

def region_adaptive_dice_bce_loss(
    pred,
    target,
    dice_weight=0.5,
    adaptive_channel="ET",
    adaptive_region="TC",
    region_multiplier=1.5,
    smooth=1.,
):
    """
    Dice + region-adaptive BCE.

    Channel order is [WT, TC, ET]. Unlike global channel weighting, this only
    increases the selected channel's BCE inside an anatomically plausible region
    from the ground-truth mask, e.g. ET errors inside TC.
    """
    channel_to_idx = {"WT": 0, "TC": 1, "ET": 2}
    if adaptive_channel not in channel_to_idx:
        raise ValueError(f"adaptive_channel must be one of {list(channel_to_idx)}, got {adaptive_channel}")
    if adaptive_region not in channel_to_idx:
        raise ValueError(f"adaptive_region must be one of {list(channel_to_idx)}, got {adaptive_region}")

    d_loss = dice_loss(pred, target, smooth=smooth)
    bce = F.binary_cross_entropy(pred, target, reduction="none")

    channel_idx = channel_to_idx[adaptive_channel]
    region_idx = channel_to_idx[adaptive_region]
    region_mask = target[:, region_idx:region_idx + 1]

    weights = torch.ones_like(bce)
    weights[:, channel_idx:channel_idx + 1] = (
        1.0 + (float(region_multiplier) - 1.0) * region_mask
    )
    bce_loss = (bce * weights).mean()
    return dice_weight * d_loss + (1.0 - dice_weight) * bce_loss

def et_positive_adaptive_dice_bce_loss(
    pred,
    target,
    dice_weight=0.5,
    positive_multiplier=1.5,
    smooth=1.,
):
    """
    Dice + BCE with extra ET positive/false-negative emphasis only.

    Channel order is [WT, TC, ET]. This avoids Exp046's issue of increasing all
    ET BCE inside TC, including the many TC-but-not-ET voxels. Only the positive
    ET term, -GT_ET * log(Pred_ET), receives the extra weight.
    """
    d_loss = dice_loss(pred, target, smooth=smooth)

    eps = 1e-7
    pred_clamped = pred.clamp(eps, 1.0 - eps)
    positive_term = -target * torch.log(pred_clamped)
    negative_term = -(1.0 - target) * torch.log(1.0 - pred_clamped)

    weights = torch.ones_like(positive_term)
    weights[:, 2:3] = float(positive_multiplier)
    bce_loss = (positive_term * weights + negative_term).mean()

    return dice_weight * d_loss + (1.0 - dice_weight) * bce_loss

def hierarchy_consistency_loss(pred):
    """
    Penalize BraTS region hierarchy violations: ET should be inside TC, TC inside WT.

    Channel order is [WT, TC, ET]. The loss stays differentiable by operating on
    probabilities instead of thresholded masks.
    """
    wt = pred[:, 0:1]
    tc = pred[:, 1:2]
    et = pred[:, 2:3]
    et_outside_tc = F.relu(et - tc).mean()
    tc_outside_wt = F.relu(tc - wt).mean()
    return et_outside_tc + tc_outside_wt

def boundary_consistency_loss(pred, target, kernel_size=3, smooth=1.):
    """
    Match soft prediction boundaries to target boundaries.

    Boundary maps are computed with a differentiable morphological gradient:
    max_pool(x) - min_pool(x). Channel order remains [WT, TC, ET].
    """
    padding = kernel_size // 2
    pred_max = F.max_pool2d(pred, kernel_size=kernel_size, stride=1, padding=padding)
    pred_min = -F.max_pool2d(-pred, kernel_size=kernel_size, stride=1, padding=padding)
    target_max = F.max_pool2d(target, kernel_size=kernel_size, stride=1, padding=padding)
    target_min = -F.max_pool2d(-target, kernel_size=kernel_size, stride=1, padding=padding)

    pred_boundary = (pred_max - pred_min).clamp(0, 1)
    target_boundary = (target_max - target_min).clamp(0, 1)
    return dice_bce_loss(pred_boundary, target_boundary, dice_weight=0.5, smooth=smooth)

def dice_focal_loss(pred, target, dice_weight=0.5, alpha=0.7, beta=0.3, gamma=4/3, smooth=1e-6):
    """
    Dice + Focal Tversky Loss (Exp012)
    - Dice Loss: giữ vững chất lượng vùng lớn (WT)
    - Focal Tversky: ưu tiên vùng nhỏ khó tìm (ET, TC)
    """
    d_loss  = dice_loss(pred, target)
    ft_loss = focal_tversky_loss(pred, target, alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
    return dice_weight * d_loss + (1.0 - dice_weight) * ft_loss

def modality_contrastive_loss(modality_features, temperature=0.07):
    """
    Contrastive-aware loss for MRI modality pairs.

    Input order: FLAIR, T1, T1ce, T2.
    Positive pairs follow DFuse-Net: FLAIR-T2 and T1-T1ce.
    """
    feats = F.normalize(modality_features, dim=-1)

    flair = feats[:, 0]
    t1 = feats[:, 1]
    t1ce = feats[:, 2]
    t2 = feats[:, 3]

    pairs = [
        (flair, t2, [t1, t1ce]),
        (t1, t1ce, [flair, t2]),
    ]

    losses = []
    for anchor, positive, negatives in pairs:
        pos = torch.exp(F.cosine_similarity(anchor, positive, dim=1) / temperature)
        neg = torch.stack([
            torch.exp(F.cosine_similarity(anchor, neg_feat, dim=1) / temperature)
            for neg_feat in negatives
        ], dim=0).sum(dim=0)
        losses.append(-torch.log(pos / (pos + neg + 1e-8)))

    return torch.stack(losses, dim=0).mean()
