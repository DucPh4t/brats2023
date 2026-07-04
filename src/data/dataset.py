# ============================================================
# File: dataset.py
# Role: Quản lý dataset BraTS2023 GLI, thực hiện cắt lát, lấy mẫu và chia split
# Dataset: ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
# File format: {sid}-t2f.nii.gz, {sid}-t1n.nii.gz, {sid}-t1c.nii.gz, {sid}-t2w.nii.gz, {sid}-seg.nii.gz
# Label mapping: 1=NCR, 2=ED, 3=ET  →  WT=(>0), TC=(1|3), ET=(==3)
# ============================================================
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import nibabel as nib
import numpy as np

from src.data.processors import get_preprocessor


class BraTSDataset(Dataset):
    """
    Unified Dataset class for BraTS2023 GLI Challenge.

    [FORMAT] Reads BraTS2023 naming: {sid}-t2f.nii.gz / -t1n / -t1c / -t2w / -seg
    [LABELS] NCR=1, ED=2, ET=3  →  WT=(mask>0), TC=(1|3), ET=(==3)
    [ABLATION] Supports: fixed / weighted / oversample sampling strategies.
    [MEMORY] cache_volumes=True pre-loads all subjects into RAM as float16.
               Set cache_volumes=False for large datasets (>100 subjects local).
    """

    # BraTS2023 modality suffix mapping → internal canonical keys
    _MOD_SUFFIXES = {
        'flair': '-t2f.nii.gz',
        't1':    '-t1n.nii.gz',
        't1ce':  '-t1c.nii.gz',
        't2':    '-t2w.nii.gz',
    }

    def __init__(self, root_dir, subject_ids, slice_range=(0, 155),
                 normalization='zscore_clip', augmentation=False,
                 augmentation_intensity=False,
                 sampling='fixed', context_slices=0, min_tumor_pixels=100,
                 cache_volumes=False, preprocess_config=None):

        self.root_dir = root_dir
        self.subject_ids = subject_ids
        self.slice_range = slice_range
        self.normalization = normalization
        self.augmentation = augmentation
        self.augmentation_intensity = augmentation_intensity
        self.sampling = sampling
        self.context_radius = max((int(context_slices) - 1) // 2, 0)
        self.min_tumor_pixels = min_tumor_pixels
        self.cache_volumes = cache_volumes

        self.preprocessor = get_preprocessor(self.normalization, preprocess_config or {})

        # ── Pre-load all volumes into RAM (float16 to save memory) ───────────
        self._cache = {}
        if self.cache_volumes:
            try:
                from tqdm import tqdm
                iter_fn = lambda x: tqdm(x, desc='Caching volumes', leave=False)
            except ImportError:
                iter_fn = lambda x: x

            print(f'  [CACHE] Pre-loading {len(subject_ids)} subjects into RAM...')
            for sid in iter_fn(subject_ids):
                self._cache[sid] = self._load_subject(sid)
        # ─────────────────────────────────────────────────────────────────────

        # Build sample list + optional weights
        self.samples = []
        self.sample_weights = []

        for sid in self.subject_ids:
            sl_start, sl_end = self.slice_range

            if self.sampling == 'weighted':
                seg_vol = self._get_seg_for_sampling(sid)
                for slice_idx in range(sl_start, sl_end):
                    self.samples.append((sid, slice_idx))
                    has_tumor = (seg_vol[:, :, slice_idx] > 0).sum() > 0
                    self.sample_weights.append(3.0 if has_tumor else 1.0)

            elif self.sampling == 'oversample':
                seg_vol = self._get_seg_for_sampling(sid)
                for slice_idx in range(sl_start, sl_end):
                    self.samples.append((sid, slice_idx))
                    if (seg_vol[:, :, slice_idx] > 0).sum() > 0:
                        self.samples.append((sid, slice_idx))  # duplicate tumor slices

            else:  # 'fixed' (default)
                for slice_idx in range(sl_start, sl_end):
                    self.samples.append((sid, slice_idx))

    # ── Private helpers ───────────────────────────────────────────────────────

    def _find_file(self, sid, suffix):
        """Locate a modality file by suffix inside subject directory.
        Handles bizarre Kaggle dataset bugs where .nii files are extracted as folders."""
        subdir = os.path.join(self.root_dir, sid)
        clean_suffix = suffix.replace('.nii.gz', '').replace('.nii', '')
        
        # 1. Direct match
        for f in os.listdir(subdir):
            if f.startswith(sid + clean_suffix):
                full_path = os.path.join(subdir, f)
                # If Kaggle incorrectly made it a folder, look inside
                if os.path.isdir(full_path):
                    inside_files = [f for f in os.listdir(full_path) if f.endswith('.nii') or f.endswith('.nii.gz')]
                    if inside_files:
                        return os.path.join(full_path, inside_files[0])
                # If it's a normal file
                elif f.endswith('.nii.gz') or f.endswith('.nii'):
                    return full_path
                    
        raise FileNotFoundError(
            f'File with suffix "{clean_suffix}" not found for subject {sid} under {subdir}'
        )

    def _load_subject(self, sid):
        """Load & normalize all 4 modalities + seg mask for one subject."""
        entry = {}
        for mod_key, suffix in self._MOD_SUFFIXES.items():
            path = self._find_file(sid, suffix)
            vol_3d = nib.load(path).get_fdata()
            entry[mod_key] = self.preprocessor(vol_3d, modality=mod_key).astype(np.float16)

        seg_path = self._find_file(sid, '-seg.nii.gz')
        entry['seg'] = nib.load(seg_path).get_fdata().astype(np.uint8)
        return entry

    def _get_seg_for_sampling(self, sid):
        """Return seg volume, from cache or disk (for weighted/oversample modes)."""
        if self.cache_volumes:
            return self._cache[sid]['seg']
        seg_path = self._find_file(sid, '-seg.nii.gz')
        return nib.load(seg_path).get_fdata().astype(np.uint8)

    def _context_indices(self, slice_idx):
        if self.context_radius == 0:
            return [slice_idx]
        return [
            int(np.clip(slice_idx + offset, 0, 154))
            for offset in range(-self.context_radius, self.context_radius + 1)
        ]

    # ── Public interface ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sid, slice_idx = self.samples[idx]

        if self.cache_volumes:
            cached = self._cache[sid]
            imgs = []
            for mod in ['flair', 't1', 't1ce', 't2']:
                for ctx_idx in self._context_indices(slice_idx):
                    imgs.append(cached[mod][:, :, ctx_idx])
            mask = cached['seg'][:, :, slice_idx]
        else:
            imgs = []
            for mod, suffix in self._MOD_SUFFIXES.items():
                path = self._find_file(sid, suffix)
                vol_3d = nib.load(path).get_fdata()
                vol_norm = self.preprocessor(vol_3d, modality=mod)
                for ctx_idx in self._context_indices(slice_idx):
                    imgs.append(vol_norm[:, :, ctx_idx])
            seg_path = self._find_file(sid, '-seg.nii.gz')
            mask = nib.load(seg_path).get_fdata()[:, :, slice_idx]

        stack = np.stack(imgs, axis=0).astype(np.float32)

        # ── BraTS2023 label encoding ──────────────────────────────────────────
        # Labels: NCR=1, ED=2, ET=3
        # WT = Whole Tumor  = all labels (1|2|3)
        # TC = Tumor Core   = NCR + ET  (1|3)  — no ED
        # ET = Enhancing    = ET only   (==3)
        wt = (mask > 0).astype(np.float32)
        tc = np.logical_or(mask == 1, mask == 3).astype(np.float32)
        et = (mask == 3).astype(np.float32)
        masks = np.stack([wt, tc, et], axis=0)

        # ── Data Augmentation (training only) ─────────────────────────────────
        if self.augmentation:
            if random.random() > 0.5:
                stack = np.flip(stack, axis=2).copy()
                masks = np.flip(masks, axis=2).copy()
            if random.random() > 0.5:
                stack = np.flip(stack, axis=1).copy()
                masks = np.flip(masks, axis=1).copy()

            if self.augmentation_intensity:
                if random.random() > 0.5:
                    stack = stack + random.uniform(-0.1, 0.1)
                if random.random() > 0.5:
                    stack = stack * random.uniform(0.9, 1.1)

        return torch.from_numpy(stack), torch.from_numpy(masks)


# ── Split Utilities ───────────────────────────────────────────────────────────

def get_subject_splits(config):
    """
    Discover and split BraTS2023 subjects from root_dir.
    Naming convention: BraTS-GLI-XXXXX-XXX

    Split: 80% train / 10% val / 10% test  (random, seed=42).
    """
    data_cfg = config['data']
    root_dir = data_cfg['root_dir']

    # Discover all valid subject directories
    raw_subjects = sorted([
        d for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith('BraTS-GLI-')
    ])
    
    # Filter out corrupted subjects (missing files or 0-byte empty files)
    subjects = []
    for sid in raw_subjects:
        subdir = os.path.join(root_dir, sid)
        is_valid = True
        # Check all 5 required modalities/masks
        for suffix in ['-t2f', '-t1n', '-t1c', '-t2w', '-seg']:
            # files can end with .nii.gz or .nii
            path_gz = os.path.join(subdir, f"{sid}{suffix}.nii.gz")
            path_nii = os.path.join(subdir, f"{sid}{suffix}.nii")
            
            # Helper to check if file exists and > 0 bytes (handles Kaggle nested folder bug)
            def check_valid(p):
                if os.path.isdir(p):
                    # It's a folder, look inside
                    for f in os.listdir(p):
                        if (f.endswith('.nii') or f.endswith('.nii.gz')) and os.path.getsize(os.path.join(p, f)) > 0:
                            return True
                    return False
                return os.path.exists(p) and os.path.getsize(p) > 0
                
            if check_valid(path_gz) or check_valid(path_nii):
                continue
            
            is_valid = False
            break
            
        if is_valid:
            subjects.append(sid)
        else:
            print(f"[WARNING] Skipping corrupted subject: {sid}")

    if len(subjects) == 0:
        raise RuntimeError(
            f'No BraTS2023 subjects found in {root_dir}. '
            f'Expected directories matching BraTS-GLI-XXXXX-XXX.'
        )

    split_type = data_cfg.get('split_type', 'random')

    if split_type == 'random':
        rng = np.random.default_rng(42)
        subjects_arr = np.array(subjects)
        rng.shuffle(subjects_arr)
        subjects = subjects_arr.tolist()
    else:
        # Sequential split (first 80% train, next 10% val, last 10% test)
        # This mirrors Exp001 in BraTS2020 to test split bias
        pass

    n = len(subjects)
    n_train = int(n * 0.80)
    n_val   = int(n * 0.10)

    train_subjects = subjects[:n_train]
    val_subjects   = subjects[n_train:n_train + n_val]
    test_subjects  = subjects[n_train + n_val:]

    print(f'[SPLIT] Total: {n} | Train: {len(train_subjects)} | '
          f'Val: {len(val_subjects)} | Test: {len(test_subjects)}')

    return train_subjects, val_subjects, test_subjects


def get_dataloaders(config):
    """Build train/val DataLoaders and return test_ds, test_subjects."""
    data_cfg  = config['data']
    train_cfg = config['training']

    train_subjects, val_subjects, test_subjects = get_subject_splits(config)

    # ── Shared dataset kwargs ─────────────────────────────────────────────────
    slice_range   = data_cfg.get('slice_range', [0, 155])
    if isinstance(slice_range, list):
        slice_range = tuple(slice_range)
    norm_type     = data_cfg.get('normalization', 'zscore_clip')
    augmentation  = data_cfg.get('augmentation', False)
    aug_intensity = data_cfg.get('augmentation_intensity', False)
    sampling      = data_cfg.get('sampling', 'fixed')
    context_slices = data_cfg.get('context_slices', 0)
    min_tumor_px  = data_cfg.get('min_tumor_pixels', 100)
    cache_volumes = data_cfg.get('cache_volumes', False)

    train_ds = BraTSDataset(
        data_cfg['root_dir'], train_subjects, slice_range,
        normalization=norm_type, augmentation=augmentation,
        augmentation_intensity=aug_intensity, sampling=sampling,
        context_slices=context_slices, min_tumor_pixels=min_tumor_px,
        cache_volumes=cache_volumes, preprocess_config=data_cfg,
    )
    # Val/Test always use fixed sampling, no augmentation, no cache override
    val_ds = BraTSDataset(
        data_cfg['root_dir'], val_subjects, (0, 155),
        normalization=norm_type, augmentation=False, sampling='fixed',
        context_slices=context_slices, cache_volumes=False,
        preprocess_config=data_cfg,
    )
    test_ds = BraTSDataset(
        data_cfg['root_dir'], test_subjects, (0, 155),
        normalization=norm_type, augmentation=False, sampling='fixed',
        context_slices=context_slices, cache_volumes=False,
        preprocess_config=data_cfg,
    )

    print(f'[DATA] Train: {len(train_ds)} slices | Val: {len(val_ds)} slices | '
          f'Test subjects: {len(test_subjects)}')

    num_workers  = train_cfg.get('num_workers', 2)
    pin_memory   = train_cfg.get('pin_memory', True)
    batch_size   = train_cfg['batch_size']

    train_sampler  = None
    shuffle_train  = True

    if (sampling == 'weighted'
            and hasattr(train_ds, 'sample_weights')
            and len(train_ds.sample_weights) == len(train_ds.samples)):
        train_sampler = WeightedRandomSampler(
            weights=train_ds.sample_weights,
            num_samples=len(train_ds.samples),
            replacement=True,
        )
        shuffle_train = False
        print('[DATA] WeightedRandomSampler enabled for train loader')

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train,
        sampler=train_sampler, num_workers=num_workers,
        pin_memory=pin_memory, persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    return train_loader, val_loader, test_ds, test_subjects
