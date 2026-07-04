import argparse
import yaml
import torch
import sys
import os

from src.data.dataset import get_dataloaders, get_subject_splits
from src.models.unet import UNet2D
from src.engine.trainer import Trainer
from src.engine.evaluator import Evaluator

import random
import numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    set_seed(42)  # Hạt giống vĩnh cửu 
    parser = argparse.ArgumentParser(description="BraTS MRI Segmentation Entrypoint")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="train", help="Mode to run")
    parser.add_argument("--resume_path", type=str, default=None, help="Path to last_checkpoint.pth to resume training")
    parser.add_argument("--stop_epoch", type=int, default=None, help="Stop early at this epoch to prevent Kaggle timeout")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        
    # --- IN RA CẤU HÌNH ĐỂ KIỂM TRA (ABLATION VERIFICATION) ---
    print("\n" + "="*50)
    print(f"🚀 RUNNING EXPERIMENT: {config.get('exp_name', 'Unknown')}")
    print(f"📝 Description: {config.get('description', '')}")
    print("-" * 50)
    print(f"📦 [DATA]")
    print(f"   - Split Type    : {config['data'].get('split_type', 'sequential')}")
    print(f"   - Sampling      : {config['data'].get('sampling', 'random')} (Slices: {config['data'].get('slice_range', 'ALL')})")
    print(f"   - Normalization : {config['data'].get('normalization', 'minmax')}")
    print(f"   - Augmentation  : {config['data'].get('augmentation', False)}")
    if config['data'].get('augmentation_intensity', False):
        print(f"   - Intensity Aug : True")
    print(f"🧠 [MODEL]")
    print(f"   - Architecture  : {config['model'].get('architecture', 'unet2d')}")
    print(f"   - Init Features : {config['model'].get('init_features', 32)}")
    print(f"⚙️  [TRAINING]")
    print(f"   - Optimizer     : {config['training'].get('optimizer', 'adam')}")
    print(f"   - Scheduler     : {config['training'].get('scheduler', 'none')}")
    print(f"   - Loss Function : {config['training'].get('loss', 'dice')}")
    if 'loss_params' in config['training']:
        print(f"   - Loss Params   : {config['training']['loss_params']}")
    print("="*50 + "\n")
        
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    init_features = config["model"].get("init_features", 32)
    arch = config["model"].get("architecture", "unet2d")
    
    if arch == "unet2d":
        model = UNet2D(init_features=init_features).to(device)
        arch_name = "U-Net 2D"
    else:
        raise ValueError(f"Architecture {arch} not supported in this clean setup yet.")

        
    num_params = sum(p.numel() for p in model.parameters())
    print(f"==========================================")
    print(f"[MODEL] Architecture: {arch_name}")
    print(f"[MODEL] Init Features: {init_features}")
    print(f"[MODEL] Total Params: {num_params:,}")
    print(f"==========================================")
    
    # Load Data Loaders
    # Khi chỉ đánh giá (eval), không cần cache 295 subjects vào RAM.
    # Evaluator tự load trực tiếp từ file → tránh OOM.
    if args.mode == "eval":
        config["data"]["cache_volumes"] = False
    if args.mode == "train":
        train_loader, val_loader, test_ds, test_subjects = get_dataloaders(config)
        trainer = Trainer(model, config, device, train_loader, val_loader, 
                          resume_path=args.resume_path, stop_epoch=args.stop_epoch)
        trainer.fit()
        
    elif args.mode == "eval":
        _, val_subjects, test_subjects = get_subject_splits(config)
        evaluator = Evaluator(model, config, device, test_subjects, val_subjects=val_subjects)
        evaluator.run()

if __name__ == "__main__":
    main()
