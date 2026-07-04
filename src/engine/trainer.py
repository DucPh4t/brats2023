import os
import torch
from tqdm import tqdm
import json
import numpy as np

from src.utils.losses import (
    dice_loss,
    focal_tversky_loss,
    dice_bce_loss,
    weighted_dice_bce_loss,
    region_adaptive_dice_bce_loss,
    et_positive_adaptive_dice_bce_loss,
    dice_focal_loss,
    boundary_consistency_loss,
    hierarchy_consistency_loss,
    modality_contrastive_loss,
)
import torch.nn.functional as F
from src.data.dataset import get_subject_splits
from src.utils.metrics import calc_dice_3d
from src.utils.postprocessing import REGION_NAMES

class Trainer:
    def __init__(self, model, config, device, train_loader, val_loader, resume_path=None, stop_epoch=None):
        self.model = model
        self.config = config
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.resume_path = resume_path
        self.stop_epoch = stop_epoch
        
        train_cfg = self.config["training"]
        
        # Setup Optimizer
        if train_cfg["optimizer"] == "adam":
            from torch.optim import Adam
            self.optimizer = Adam(self.model.parameters(), lr=train_cfg["lr"])
        elif train_cfg["optimizer"] == "adamw":
            from torch.optim import AdamW
            self.optimizer = AdamW(
                self.model.parameters(), 
                lr=train_cfg["lr"], 
                weight_decay=train_cfg.get("weight_decay", 1e-5)
            )
            
        self.epochs = train_cfg["epochs"]
        
        # Output paths
        self.exp_name = config["exp_name"]
        self.out_dir  = os.path.join("outputs", self.exp_name)
        os.makedirs(self.out_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.out_dir, "best_model.pth")
        self.best_model_3d_path = os.path.join(self.out_dir, "best_model_3d.pth")
        
        # Setup Scheduler
        self.scheduler = None
        if "scheduler" in train_cfg:
            if train_cfg["scheduler"] == "plateau":
                from torch.optim.lr_scheduler import ReduceLROnPlateau
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)
            elif train_cfg["scheduler"] == "cosine":
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs)
            elif train_cfg["scheduler"] == "polynomial":
                from torch.optim.lr_scheduler import LambdaLR
                power = train_cfg.get("scheduler_power", 0.9)
                self.scheduler = LambdaLR(
                    self.optimizer,
                    lr_lambda=lambda e: (1 - e/self.epochs) ** power
                )
        # Thêm logic load Resume Checkpoint
        self.start_epoch = 0
        self.best_val_dice = 0.0
        self.best_val3d_score = None
        self.history = {"train_loss": [], "val_loss": [], "val_dice": []}
        self.validation_3d_cfg = self.config.get("validation_3d", {})
        self.val3d_subjects = []
        if self.validation_3d_cfg.get("enabled", False):
            _, self.val3d_subjects, _ = get_subject_splits(self.config)
            max_subjects = self.validation_3d_cfg.get("max_subjects")
            if max_subjects:
                self.val3d_subjects = self.val3d_subjects[: int(max_subjects)]
            self.history.setdefault("val3d", [])
        
        if self.resume_path and os.path.exists(self.resume_path):
            print(f"\n🔄 RESUMING FROM CHECKPOINT: {self.resume_path}")
            checkpoint = torch.load(self.resume_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if self.scheduler and checkpoint.get('scheduler_state'):
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_dice = checkpoint.get('best_val_dice', 0.0)
            self.best_val3d_score = checkpoint.get('best_val3d_score')
            self.history = checkpoint.get('history', {"train_loss": [], "val_loss": [], "val_dice": []})
            if self.validation_3d_cfg.get("enabled", False):
                self.history.setdefault("val3d", [])
            
            # Khôi phục luôn file best_model.pth cũ (đề phòng Hiệp 2 không phá được kỷ lục)
            import shutil
            old_best_model = os.path.join(os.path.dirname(self.resume_path), "best_model.pth")
            if os.path.exists(old_best_model):
                shutil.copy(old_best_model, self.best_model_path)
            old_best_model_3d = os.path.join(os.path.dirname(self.resume_path), "best_model_3d.pth")
            if os.path.exists(old_best_model_3d):
                shutil.copy(old_best_model_3d, self.best_model_3d_path)
                
            print(f"   [+] Resumed successfully! Starting from Epoch {self.start_epoch+1}")
            print(f"   [+] Best Val Dice so far: {self.best_val_dice:.4f}\n")
            
    def _compute_loss(self, preds, masks):
        loss_type = self.config["training"]["loss"]
        loss_kwargs = dict(self.config["training"].get("loss_params", {}))
        hierarchy_weight = float(loss_kwargs.pop("hierarchy_weight", 0.0))
        boundary_weight = float(loss_kwargs.pop("boundary_weight", 0.0))
        boundary_kernel_size = int(loss_kwargs.pop("boundary_kernel_size", 3))
        boundary_channel = loss_kwargs.pop("boundary_channel", None)

        if loss_type == "bce":
            loss = F.binary_cross_entropy(preds, masks)
        elif loss_type == "focal_tversky":
            loss = focal_tversky_loss(preds, masks, **loss_kwargs)
        elif loss_type == "dice_bce":
            loss = dice_bce_loss(preds, masks, **loss_kwargs)
        elif loss_type == "weighted_dice_bce":
            loss = weighted_dice_bce_loss(preds, masks, **loss_kwargs)
        elif loss_type == "region_adaptive_dice_bce":
            loss = region_adaptive_dice_bce_loss(preds, masks, **loss_kwargs)
        elif loss_type == "et_positive_adaptive_dice_bce":
            loss = et_positive_adaptive_dice_bce_loss(preds, masks, **loss_kwargs)
        elif loss_type == "dice_focal":
            loss = dice_focal_loss(preds, masks, **loss_kwargs)
        else:  # default: dice
            loss = dice_loss(preds, masks, **loss_kwargs)

        if hierarchy_weight > 0:
            loss = loss + hierarchy_weight * hierarchy_consistency_loss(preds)
        if boundary_weight > 0:
            boundary_preds, boundary_masks = preds, masks
            if boundary_channel is not None:
                channel_to_idx = {"WT": 0, "TC": 1, "ET": 2}
                if boundary_channel not in channel_to_idx:
                    raise ValueError(
                        f"boundary_channel must be one of {list(channel_to_idx)}, got {boundary_channel}"
                    )
                channel_idx = channel_to_idx[boundary_channel]
                boundary_preds = preds[:, channel_idx:channel_idx + 1]
                boundary_masks = masks[:, channel_idx:channel_idx + 1]
            loss = loss + boundary_weight * boundary_consistency_loss(
                boundary_preds,
                boundary_masks,
                kernel_size=boundary_kernel_size,
            )
        return loss

    def _compute_deep_supervision_loss(self, aux, masks):
        ds_cfg = self.config["training"].get("deep_supervision", {})
        if not ds_cfg.get("enabled", False):
            return None
        outputs = aux.get("deep_supervision") if isinstance(aux, dict) else None
        if not outputs:
            return None

        weights = ds_cfg.get("weights", [0.5, 0.25, 0.125])
        if len(weights) != len(outputs):
            raise ValueError(
                f"deep_supervision.weights must have {len(outputs)} values, got {len(weights)}"
            )

        total_weight = float(sum(weights))
        if total_weight <= 0:
            raise ValueError("deep_supervision.weights must sum to a positive value")

        ds_loss = 0.0
        for output, weight in zip(outputs, weights):
            aux_masks = F.interpolate(masks, size=output.shape[-2:], mode="nearest")
            aux_probs = torch.sigmoid(output)
            ds_loss = ds_loss + (float(weight) / total_weight) * self._compute_loss(aux_probs, aux_masks)
        return ds_loss

    def _unpack_model_output(self, output):
        if isinstance(output, tuple):
            return output
        return output, {}

    def _compute_contrastive_loss(self, aux):
        contrastive_cfg = self.config["training"].get("contrastive", {})
        if not contrastive_cfg.get("enabled", False):
            return None
        if "modality_features" not in aux:
            return None
        return modality_contrastive_loss(
            aux["modality_features"],
            temperature=contrastive_cfg.get("temperature", 0.07),
        )

    def _compute_et_presence_loss(self, aux, masks):
        presence_cfg = self.config["training"].get("et_presence", {})
        if not presence_cfg.get("enabled", False):
            return None
        if "et_presence_logit" not in aux:
            return None
        et_present = (masks[:, 2].amax(dim=(1, 2)) > 0).float().unsqueeze(1)
        return F.binary_cross_entropy_with_logits(aux["et_presence_logit"], et_present)

    def _should_run_3d_validation(self, epoch):
        if not self.validation_3d_cfg.get("enabled", False):
            return False
        interval = int(self.validation_3d_cfg.get("interval_epochs", 5))
        return (epoch + 1) % interval == 0 or (epoch + 1) == self.epochs

    def _val3d_score(self, summary):
        monitor = self.validation_3d_cfg.get("monitor", "mean_dice")
        if monitor == "mean_dice":
            return summary["DICE"]["Mean"]
        if monitor == "et_hd95":
            if "ET" not in summary.get("HD95", {}):
                raise ValueError("validation_3d.monitor='et_hd95' requires validation_3d.include_hd95=true")
            return -summary["HD95"]["ET"]
        raise ValueError(f"Unknown validation_3d.monitor: {monitor}")

    def _run_3d_validation(self, epoch):
        metrics = self._evaluate_val_loader_3d(epoch)
        summary = metrics["summary"]
        score = self._val3d_score(summary)
        record = {
            "epoch": epoch + 1,
            "monitor": self.validation_3d_cfg.get("monitor", "mean_dice"),
            "score": score,
            "dice": summary["DICE"],
            "hd95": summary.get("HD95", {}),
            "num_subjects": metrics["num_subjects"],
        }
        self.history.setdefault("val3d", []).append(record)
        msg = f"   [3D Val] Mean Dice: {summary['DICE']['Mean']:.2f}%"
        if "ET" in summary.get("HD95", {}):
            msg += f" | ET-HD95: {summary['HD95']['ET']:.2f}"
        print(msg)

        if self.best_val3d_score is None or score > self.best_val3d_score:
            self.best_val3d_score = score
            torch.save(self.model.state_dict(), self.best_model_3d_path)
            print(f"   [+] 3D validation improved! Saved best_model_3d.pth")

    def _evaluate_val_loader_3d(self, epoch):
        """Stream validation slices into subject volumes without reloading NIfTI files."""
        thresholds = self.validation_3d_cfg.get(
            "thresholds",
            {"WT": 0.5, "TC": 0.5, "ET": 0.5},
        )
        threshold_tensor = torch.tensor(
            [thresholds[region] for region in REGION_NAMES],
            dtype=torch.float32,
            device=self.device,
        ).view(1, 3, 1, 1)
        subject_filter = set(self.val3d_subjects)
        samples = self.val_loader.dataset.samples

        per_subject = []
        current_sid = None
        pred_slices, gt_slices = [], []
        sample_offset = 0

        def flush_subject():
            if current_sid is None or not pred_slices:
                return
            pred_volume = np.stack(pred_slices, axis=0)
            gt_volume = np.stack(gt_slices, axis=0)
            per_subject.append(self._score_3d_subject(current_sid, pred_volume, gt_volume))

        self.model.eval()
        with torch.no_grad():
            iterator = tqdm(self.val_loader, desc=f"Epoch {epoch + 1} [3D Val]", leave=False)
            for imgs, masks in iterator:
                batch_size = imgs.shape[0]
                sample_meta = samples[sample_offset: sample_offset + batch_size]
                sample_offset += batch_size

                imgs = imgs.to(self.device)
                logits, _ = self._unpack_model_output(self.model(imgs))
                probs = torch.sigmoid(logits)
                binary = (probs > threshold_tensor).cpu().numpy().astype(np.float32)
                gt_batch = masks.numpy().astype(np.float32)

                for item_idx, (sid, _) in enumerate(sample_meta):
                    if subject_filter and sid not in subject_filter:
                        continue
                    if current_sid is None:
                        current_sid = sid
                    if sid != current_sid:
                        flush_subject()
                        current_sid = sid
                        pred_slices, gt_slices = [], []
                    pred_slices.append(binary[item_idx])
                    gt_slices.append(gt_batch[item_idx])

        flush_subject()
        return {
            "summary": self._summarize_3d_subjects(per_subject),
            "per_subject": per_subject,
            "num_subjects": len(per_subject),
        }

    def _score_3d_subject(self, subject_id, pred_volume, gt_volume):
        scores = {"subject_id": subject_id, "DICE": {}, "HD95": {}}
        include_hd95 = bool(self.validation_3d_cfg.get("include_hd95", False))
        for channel, region in enumerate(REGION_NAMES):
            pred = pred_volume[:, channel]
            gt = gt_volume[:, channel]
            scores["DICE"][region] = float(calc_dice_3d(pred, gt))
            if include_hd95:
                from src.utils.metrics import calc_hd95_3d
                scores["HD95"][region] = float(calc_hd95_3d(pred, gt, voxelspacing=(1.0, 1.0, 1.0)))
        return scores

    def _summarize_3d_subjects(self, per_subject):
        summary = {"DICE": {}, "HD95": {}}
        metrics = ("DICE", "HD95") if self.validation_3d_cfg.get("include_hd95", False) else ("DICE",)
        for metric in metrics:
            for region in REGION_NAMES:
                values = np.array([item[metric][region] for item in per_subject], dtype=np.float32)
                multiplier = 100.0 if metric == "DICE" else 1.0
                summary[metric][region] = round(float(values.mean() * multiplier), 2)
                summary[metric][f"{region}_std"] = round(float(values.std() * multiplier), 2)
                summary[metric][f"{region}_median"] = round(float(np.median(values) * multiplier), 2)
        summary["DICE"]["Mean"] = round(float(np.mean([summary["DICE"][r] for r in REGION_NAMES])), 2)
        if summary["HD95"]:
            summary["HD95"]["Mean"] = round(float(np.mean([summary["HD95"][r] for r in REGION_NAMES])), 2)
        return summary

    def fit(self):
        for epoch in range(self.start_epoch, self.epochs):
            # --- TRAIN ---
            self.model.train()
            train_loss = 0.0
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs} [Train]")
            for imgs, masks in pbar:
                imgs, masks = imgs.to(self.device), masks.to(self.device)
                
                self.optimizer.zero_grad()
                preds, aux = self._unpack_model_output(self.model(imgs))
                probs = torch.sigmoid(preds) # [PAPER FIX] Apply sigmoid trước loss (nhận probs, không phải logits)
                loss  = self._compute_loss(probs, masks)
                deep_supervision_loss = self._compute_deep_supervision_loss(aux, masks)
                if deep_supervision_loss is not None:
                    weight = self.config["training"]["deep_supervision"].get("weight", 0.3)
                    loss = loss + weight * deep_supervision_loss
                contrastive_loss = self._compute_contrastive_loss(aux)
                if contrastive_loss is not None:
                    weight = self.config["training"]["contrastive"].get("weight", 0.1)
                    loss = loss + weight * contrastive_loss
                et_presence_loss = self._compute_et_presence_loss(aux, masks)
                if et_presence_loss is not None:
                    weight = self.config["training"]["et_presence"].get("weight", 0.1)
                    loss = loss + weight * et_presence_loss
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
            avg_train = train_loss / len(self.train_loader)
            
            # --- VAL ---
            self.model.eval()
            val_loss = 0.0
            val_dice_wt, val_dice_tc, val_dice_et = [], [], []

            with torch.no_grad():
                for imgs, masks in self.val_loader:
                    imgs, masks = imgs.to(self.device), masks.to(self.device)
                    preds, aux = self._unpack_model_output(self.model(imgs))
                    probs = torch.sigmoid(preds) # [PAPER FIX] Apply sigmoid trước loss
                    loss  = self._compute_loss(probs, masks)
                    deep_supervision_loss = self._compute_deep_supervision_loss(aux, masks)
                    if deep_supervision_loss is not None:
                        weight = self.config["training"]["deep_supervision"].get("weight", 0.3)
                        loss = loss + weight * deep_supervision_loss
                    contrastive_loss = self._compute_contrastive_loss(aux)
                    if contrastive_loss is not None:
                        weight = self.config["training"]["contrastive"].get("weight", 0.1)
                        loss = loss + weight * contrastive_loss
                    et_presence_loss = self._compute_et_presence_loss(aux, masks)
                    if et_presence_loss is not None:
                        weight = self.config["training"]["et_presence"].get("weight", 0.1)
                        loss = loss + weight * et_presence_loss
                    val_loss += loss.item()
                    
                    # [DESIGN] Tính slice-level Dice để monitor — không phải 3D volume eval
                    # Đây chỉ dùng để save best model, không phải final metric
                    binary = (probs > 0.5).float()
                    for i, dice_list in enumerate([val_dice_wt, val_dice_tc, val_dice_et]):
                        inter = (binary[:, i] * masks[:, i]).sum()
                        denom = binary[:, i].sum() + masks[:, i].sum()
                        dice  = (2 * inter / denom).item() if denom > 0 else 1.0
                        dice_list.append(dice)
                        
            avg_val   = val_loss / len(self.val_loader)
            mean_dice = (np.mean(val_dice_wt) + np.mean(val_dice_tc) + np.mean(val_dice_et)) / 3

            print(f"-> Epoch {epoch+1:02d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Mean Dice: {mean_dice:.4f}")
            
            # Step Scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(mean_dice)
                else:
                    self.scheduler.step()
            
            self.history["train_loss"].append(avg_train)
            self.history["val_loss"].append(avg_val)
            self.history["val_dice"].append(mean_dice)
            
            # [DESIGN] Save best model theo mean val Dice, không phải val loss
            if mean_dice > self.best_val_dice:
                self.best_val_dice = mean_dice
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"   [+] Mean Dice improved to {mean_dice:.4f}! Saved best_model.pth")

            if self._should_run_3d_validation(epoch):
                self._run_3d_validation(epoch)
                
            # [DESIGN] Save last_checkpoint mỗi epoch để chống Kaggle timeout
            checkpoint_state = {
                'epoch': epoch,
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_dice': self.best_val_dice,
                'best_val3d_score': self.best_val3d_score,
                'history': self.history
            }
            torch.save(checkpoint_state, os.path.join(self.out_dir, "last_checkpoint.pth"))
            
            # Kiểm tra Stop Epoch an toàn
            if self.stop_epoch and (epoch + 1) == self.stop_epoch:
                print(f"\n✋ CHỦ ĐỘNG DỪNG SỚM TẠI EPOCH {epoch+1} (Để tránh Kaggle Timeout).")
                print("   Dùng cờ --resume_path cho lần chạy sau để tiếp tục!")
                break
            
        # Save training history
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=4)
        
        print("\nTraining complete!")
