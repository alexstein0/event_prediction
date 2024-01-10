import logging
import os

import torch
import transformers
from einops import rearrange
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Tuple

log = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        model: transformers.PreTrainedModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        if cfg.loss_fn == "CrossEntropyLoss":
            self.loss_fn = CrossEntropyLoss()
        else:
            raise ValueError(f"Expected 'CrossEntropyLoss' but got {cfg.loss_fn}")

        if cfg.optim == "AdamW":
            self.optim = AdamW(model.parameters(), lr=cfg.lr)
        else:
            raise ValueError(f"Expected 'AdamW' but got {cfg.optim}")

        self.epochs = cfg.epochs
        self.chkpt_path = cfg.chkpt_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def train(self):
        # TODO:
        # 1. logging (see wandb from CV class project)
        # 2. Add perplexity and/or accuracy metric (see huggingface example)
        # 3. gradient accumulation: https://huggingface.co/learn/nlp-course/chapter7/6#training-with-accelerate
        # 4. Look at the optimizer and hyperparameters used in the hugging face example
        # 5. Add ability to start training from a checkpoint (CV class project)
        # 6. [only if needed ] Use Lighting or Huggingface Accelerate for multi-gpu (see CV class project)
        for epoch in range(self.epochs):
            self.model.train()
            for step, batch in enumerate(self.train_loader, start=1):
                self.optim.zero_grad()
                loss = self._get_loss(batch)
                loss.backward()
                self.optim.step()
                # if step % 100 == 0:
                log.info(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item()}")

        self._validate()
        save_checkpoint(self.model, self.chkpt_path, epoch)
        return self.chkpt_path


    def _validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                loss = self._get_loss(batch)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        log.info(f"Validation loss: {avg_loss}")


    def _get_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        batch = (inputs, targets)
        inputs shape: (b, t)
        targets shape: (b, t)
        logits shape: (b, t, v)
        b: batch size
        t: sequence length (tokens)
        v: vocab size
        """
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        logits = self.model(inputs).logits
        logits = rearrange(logits, 'b t v -> (b t) v')  # (b, n, v) -> (b*n, v)
        targets = rearrange(targets, 'b t -> (b t)')  # (b, n) -> (b*n)
        loss = self.loss_fn(logits, targets)
        return loss


def get_trainer(
    cfg: DictConfig,
    model: transformers.PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    return Trainer(cfg, model, train_loader, val_loader)


def save_checkpoint(model, chkpt_dir: str, epoch: int = None) -> str:
    """Save model weights to disk."""
    #TODO: save optimizer and scheduler state dicts as well so we can resume training
    # Something like:
    # checkpoint = {
    #     "model": model.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    #     "scheduler": scheduler.state_dict(),
    #     "epoch": epoch,
    #     "loss": loss,
    # }
    chkpt_dir = os.path.join(get_original_cwd(), chkpt_dir)
    filepath = os.path.join(chkpt_dir, f"epoch_{epoch}_chkpt.pth")
    os.makedirs(chkpt_dir, exist_ok=True)
    torch.save(model.state_dict(), filepath)
    return filepath