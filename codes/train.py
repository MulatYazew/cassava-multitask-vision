"""
AgroVision Trainer
Training loop with:
  - Weighted CrossEntropyLoss (handles class imbalance)
  - CosineAnnealingLR scheduler
  - Early stopping on validation loss
  - Best-model checkpointing
  - MPS (Apple Silicon) compatible
"""

import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class Trainer:
    """
    Manages the full training loop.

    Args:
        model         : PyTorch model.
        device        : torch.device.
        learning_rate : Initial LR (default 1e-3).
        weight_decay  : AdamW weight decay (default 1e-5).
        class_weights : Optional 1-D tensor for weighted loss (shape: num_classes).
                        Pass compute_class_weights(train_df) from data_handler.
    """

    def __init__(
        self,
        model:          nn.Module,
        device:         torch.device,
        learning_rate:  float ,
        weight_decay:   float ,
        class_weights:  Optional[torch.Tensor] = None,
    ) -> None:
        self.model  = model
        self.device = device

        # Weighted loss counteracts class imbalance.
        w = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=w)

        # AdamW is generally preferred over Adam for regularization.
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        self.history: dict[str, list[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc":  [], "val_acc":  [],
            "lr":         [],
        }
        self._best_val_loss    = float("inf")
        self._patience_counter = 0

    #  Single-epoch helpers 

    def run_epoch(
        self, loader: torch.utils.data.DataLoader, training: bool
    ) -> tuple[float, float]:
        self.model.train(training)
        total_loss, correct, n = 0.0, 0, 0

        ctx = torch.enable_grad() if training else torch.no_grad()
        with ctx:
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)

                if training:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss    = self.criterion(outputs, labels)

                if training:
                    loss.backward()
                    # Gradient clipping improves MPS stability.
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                correct    += (outputs.argmax(1) == labels).sum().item()
                n          += labels.size(0)

        return total_loss / n, correct / n

    #  Full training loop 

    def train(
        self,
        train_loader:   torch.utils.data.DataLoader,
        val_loader:     torch.utils.data.DataLoader,
        num_epochs:     int ,
        patience:       int ,
        model_save_dir: str,
    ) -> dict[str, list[float]]:
        """
        Train with early stopping and cosine LR annealing.

        Args:
            train_loader   : Training DataLoader.
            val_loader     : Validation DataLoader.
            num_epochs     : Maximum epochs.
            patience       : Early-stopping patience (epochs without improvement).
            model_save_dir : Directory to save best_model.pth.

        Returns:
            history dict with train/val loss & accuracy per epoch, plus LR.
        """
        Path(model_save_dir).mkdir(parents=True, exist_ok=True)
        scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        t0 = time.time()
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self._run_epoch(train_loader, training=True)
            val_loss,   val_acc   = self._run_epoch(val_loader,   training=False)
            scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(lr)

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train loss {train_loss:.4f}  acc {train_acc:.4f} | "
                f"Val loss {val_loss:.4f}  acc {val_acc:.4f} | "
                f"LR {lr:.2e}"
            )

            #  Checkpoint & early stopping 
            if val_loss < self._best_val_loss:
                self._best_val_loss    = val_loss
                self._patience_counter = 0
                ckpt = Path(model_save_dir) / "best_model.pth"
                torch.save(self.model.state_dict(), ckpt)
                print(f"  ✓ Best model saved → {ckpt}")
            else:
                self._patience_counter += 1
                if self._patience_counter >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}.")
                    break

        elapsed = time.time() - t0
        print(f"\nTraining finished in {elapsed / 60:.1f} min.")
        return self.history