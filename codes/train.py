import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score


class Trainer:
    """
    Manages the full training loop.

    Args:
        model         : PyTorch model.
        device        : torch.device.
        learning_rate : Initial LR.
        weight_decay  : AdamW weight decay.
        class_weights : Optional 1-D tensor for weighted loss (shape: num_classes).
        criterion     : Optional pre-built loss module (overrides class_weights).
        mixup_alpha   : Beta distribution alpha for MixUp. 0.0 = disabled.
        cutmix_alpha  : Beta distribution alpha for CutMix. 0.0 = disabled.
                        When both > 0, one is chosen randomly per batch.
    """

    def __init__(
        self,
        model:          nn.Module,
        device:         torch.device,
        learning_rate:  float,
        weight_decay:   float,
        class_weights:  Optional[torch.Tensor] = None,
        criterion:      Optional[nn.Module]    = None,
        mixup_alpha:    float = 0.0,
        cutmix_alpha:   float = 0.0,
    ) -> None:
        self.model        = model
        self.device       = device
        self.mixup_alpha  = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

        if criterion is not None:
            self.criterion = criterion
        else:
            w = class_weights.to(device) if class_weights is not None else None
            self.criterion = nn.CrossEntropyLoss(weight=w)

        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.history: dict[str, list[float]] = {
            "train_loss":    [],
            "val_loss":      [],
            "train_acc":     [],
            "val_acc":       [],
            "val_macro_f1":  [],
            "lr":            [],
        }
        self.best_val_f1      = -1.0
        self.patience_counter = 0

    #  MixUp / CutMix helpers

    def mixup(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """MixUp: linear interpolation of image pairs and their labels."""
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        idx = torch.randperm(images.size(0), device=self.device)
        mixed = lam * images + (1.0 - lam) * images[idx]
        return mixed, labels, labels[idx], lam

    def cutmix(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """CutMix: paste a rectangular patch from one image into another."""
        lam = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
        idx = torch.randperm(images.size(0), device=self.device)

        _, _, H, W = images.shape
        cut_rat = np.sqrt(1.0 - lam)
        cut_w   = int(W * cut_rat)
        cut_h   = int(H * cut_rat)
        cx      = np.random.randint(W)
        cy      = np.random.randint(H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)

        mixed = images.clone()
        mixed[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / (H * W)
        return mixed, labels, labels[idx], lam

    def apply_mix(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Randomly choose between MixUp and CutMix (or the enabled one)."""
        both = self.mixup_alpha > 0 and self.cutmix_alpha > 0
        use_cutmix = (self.cutmix_alpha > 0 and self.mixup_alpha == 0) or (both and np.random.random() < 0.5)
        return self.cutmix(images, labels) if use_cutmix else self.mixup(images, labels)

    #  Single-epoch helpers

    def run_epoch(
        self, loader: torch.utils.data.DataLoader, training: bool
    ) -> tuple[float, float, float]:
        """
        Run one epoch; return (avg_loss, accuracy, macro_f1).

        macro_f1 is computed for validation only (not training — too slow and
        not needed for checkpointing).  Returns -1.0 during training.

        When MixUp/CutMix is enabled during training, accuracy is computed
        against the dominant label (labels_a) as an approximation.
        """
        self.model.train(training)
        total_loss, correct, n = 0.0, 0, 0
        all_preds, all_labels = [], []

        use_mix = training and (self.mixup_alpha > 0 or self.cutmix_alpha > 0)

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                if use_mix:
                    images, labels_a, labels_b, lam = self.apply_mix(images, labels)
                    outputs = self.model(images)
                    loss = lam * self.criterion(outputs, labels_a) + (1.0 - lam) * self.criterion(outputs, labels_b)
                    preds   = outputs.argmax(1)
                    correct += (
                        lam       * (preds == labels_a).float() +
                        (1 - lam) * (preds == labels_b).float()
                    ).sum().item()
                else:
                    outputs = self.model(images)
                    loss    = self.criterion(outputs, labels)
                    preds   = outputs.argmax(1)
                    correct += (preds == labels).sum().item()

                if training:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                n          += labels.size(0)

                if not training:
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

        avg_loss = total_loss / n
        accuracy = correct / n
        macro_f1 = (
            f1_score(all_labels, all_preds, average="macro", zero_division=0)
            if not training
            else -1.0
        )
        return avg_loss, accuracy, macro_f1

    #  Full training loop

    def train(
        self,
        train_loader:   torch.utils.data.DataLoader,
        val_loader:     torch.utils.data.DataLoader,
        num_epochs:     int,
        patience:       int,
        model_save_dir: str,
    ) -> dict[str, list[float]]:
        """
        Train with early stopping (on macro-F1) and cosine LR annealing.

        Args:
            train_loader   : Training DataLoader.
            val_loader     : Validation DataLoader.
            num_epochs     : Maximum epochs.
            patience       : Early-stopping patience (epochs without macro-F1 gain).
            model_save_dir : Directory to save best_model.pth.

        Returns:
            history dict: train/val loss, accuracy, val_macro_f1, lr per epoch.
        """
        Path(model_save_dir).mkdir(parents=True, exist_ok=True)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        t0 = time.time()
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc, _            = self.run_epoch(train_loader, training=True)
            val_loss,   val_acc,   val_macro_f1 = self.run_epoch(val_loader,   training=False)
            scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_macro_f1"].append(val_macro_f1)
            self.history["lr"].append(lr)

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train loss {train_loss:.4f}  acc {train_acc:.4f} | "
                f"Val loss {val_loss:.4f}  acc {val_acc:.4f}  macro-F1 {val_macro_f1:.4f} | "
                f"LR {lr:.2e}"
            )

            if val_macro_f1 > self.best_val_f1:
                self.best_val_f1      = val_macro_f1
                self.patience_counter = 0
                ckpt = Path(model_save_dir) / "best_model.pth"
                torch.save(self.model.state_dict(), ckpt)
                print(f"  ✓ Best model saved (macro-F1={val_macro_f1:.4f}) → {ckpt}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    print(f"\nEarly stopping at epoch {epoch} — no macro-F1 gain for {patience} epochs.")
                    break

        elapsed = time.time() - t0
        print(f"\nTraining finished in {elapsed / 60:.1f} min.")
        return self.history
