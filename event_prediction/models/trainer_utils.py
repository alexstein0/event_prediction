import logging
import os
from typing import Dict, Tuple

import lightning as L
import torch
import transformers
from einops import rearrange
from hydra.utils import get_original_cwd
from lightning.pytorch import callbacks
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader


class LightingWrapper(L.LightningModule):
    """
    Wraps a torch.nn.Module or compatible huggingface model in a LightningModule class for automated training.
    A LightningModule tightly couples the model with a an optimizer and train step specifications. This allows
    us to call ".fit()" on the model and have it train with some ideal default settings. The automation we get is:
    - logging (locally or to wandb)
    - checkpointing with early stopping and for resuming traning
    - gradient accumulation
    - mixed precision training
    - multi-gpu training
    - boilerplate (.zerograd(), .eval(), .to(device), etc.)
    - does some sanity checking for us with decent error messages
    The biggest drawback is that checkpoints are saved as a LightningModule dict, so have to be smart about extracting
    the original model weights from the checkpoint dict.
    """

    def __init__(self, model, loss_fn, optim, lr):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.lr = lr

    def forward(self, X):
        """Required LightningModule method."""
        return self.model(X)
    
    def configure_optimizers(self):
        """Required LightningModule method."""
        return self.optim

    def training_step(self, batch, batch_idx):
        """Required LightningModule method. 
        Gets called from .fit() and automatically handles .to(device), .zero_grad(), .backward(), and .step()"""
        loss = self.get_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Required LightningModule method.
        Gets called from .fit() and automatically handles .eval() and calls mentioned in training_step()"""
        loss = self.get_loss(batch)
        # Lightning automatically computes the average of this loss over all the steps in the epoch
        self.log("val_loss", loss, prog_bar=True) 
        # TODO: add perplexity and/or accuracy metric 
        # preds = torch.argmax(logits, dim=1)
        # acc = (preds == y).float().mean()
        # self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def get_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Helper function for computing loss.

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
    
    @staticmethod
    def extract_statedict(ckpt_path: str) -> Dict:
        """Helper function for extracting the statedict from a LightningModule checkpoint."""
        checkpoint = torch.load(ckpt_path)
        lightning_statedict = checkpoint["state_dict"]
        statedict = {k.replace("model.", ""): v for k, v in lightning_statedict.items()}
        return statedict
    
class Trainer:
    def __init__(
        self,
        cfg: DictConfig,
        model: transformers.PreTrainedModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
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

        ckpt_dir = os.path.join(get_original_cwd(), cfg.ckpt_path)

        self.wrapped_model = LightingWrapper(model, self.loss_fn, self.optim, cfg.lr)
        self.ckpt = callbacks.ModelCheckpoint(dirpath=ckpt_dir)
        self.wandb_logger = L.pytorch.loggers.WandbLogger(project=cfg.project, log_model=cfg.upload_ckpt)
        self.trainer = L.Trainer(
            max_epochs=cfg.epochs,
            val_check_interval=1 / cfg.val_checks_per_epoch,
            accelerator="auto",
            callbacks=[self.ckpt],
            logger=self.wandb_logger,
        )


    def train(self):
        # TODO:
        # 1. logging (see wandb from CV class project)
        # 2. Add perplexity and/or accuracy metric (see huggingface example)
        # 3. gradient accumulation: https://huggingface.co/learn/nlp-course/chapter7/6#training-with-accelerate
        # 4. Look at the optimizer and hyperparameters used in the hugging face example
        # 5. Add ability to start training from a checkpoint (CV class project)
        # 6. [only if needed ] Use Lighting or Huggingface Accelerate for multi-gpu (see CV class project)
        self.trainer.fit(self.wrapped_model, self.train_loader, self.val_loader)
        self.wandb_logger.experiment.finish()
        statedict_path = self.save_statedict(self.ckpt.best_model_path)
        return statedict_path


    def save_statedict(self, ckpt_path):
        """Extract and save the statedict from a PyTorch Lightning checkpoint."""        
        statedict = LightingWrapper.extract_statedict(ckpt_path)
        statedict_path = ckpt_path.replace(".ckpt", ".pth")
        torch.save(statedict, statedict_path)
        return statedict_path


def get_trainer(
    cfg: DictConfig,
    model: transformers.PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    return Trainer(cfg, model, train_loader, val_loader)