import logging
import os
from typing import Dict, Tuple

import lightning as L
import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from hydra.utils import get_original_cwd
from lightning.pytorch import callbacks
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

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

    def __init__(self, model, loss_fn, optim, lr_scheduler, grad_accum_batches=1, tokens_per_trans=16, isfraud_token_id=157, notfraud_token_id=2):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.grad_accum_batches = grad_accum_batches
        self.tokens_per_trans = tokens_per_trans
        self.isfraud_token_id = isfraud_token_id
        self.notfraud_token_id = notfraud_token_id

    def forward(self, X):
        """Required LightningModule method."""
        return self.model(X)
    
    def configure_optimizers(self):
        """Required LightningModule method."""
        lr_sched_cfg = {
            "scheduler": self.lr_scheduler,
            "interval": "step",
            "frequency": self.grad_accum_batches,
        }
        return {"optimizer": self.optim, "lr_scheduler": self.lr_scheduler}

    def training_step(self, batch, batch_idx):
        """Required LightningModule method. 
        Gets called from .fit() and automatically handles .to(device), .zero_grad(), .backward(), and .step()"""
        loss = self.get_loss(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Required LightningModule method.
        Gets called from .fit() and automatically handles .eval() and calls mentioned in training_step()"""
        inputs, targets = batch
        logits = self.model(inputs).logits
        # CrossEntropyLoss was written for classical classification using batch_size x num_classes, so we flatten the token dimension into the batch dimension
        logits = rearrange(logits, 'b n v -> (b n) v')  # (b, n, v) -> (b*n, v)
        targets = rearrange(targets, 'b n -> (b n)')  # (b, n) -> (b*n)
        loss = self.loss_fn(logits, targets)
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")

        # Select only the last token in each transaction to evaluate as a prediction
        selected_logits = logits[self.tokens_per_trans - 1 : : self.tokens_per_trans]  # (b*n, vocab_size) -> (b*n/tokens_per_trans, vocab_size)
        selected_targets = targets[self.tokens_per_trans - 1 : : self.tokens_per_trans]
        
        # Convert logits over all vocabulary to fraud/not-fraud probability for use in accuracy
        notfraud_logits = selected_logits[:, self.notfraud_token_id]
        isfraud_logits = selected_logits[:, self.isfraud_token_id] # (b*n/tokens_per_trans, vocab_size) -> (b*n/tokens_per_trans)
        fraud_logits = torch.stack((notfraud_logits, isfraud_logits))
        fraud_probs = F.softmax(fraud_logits, dim=0)

        # Calculate AUC
        try:
            binary_targets = selected_targets == self.isfraud_token_id
            auc = roc_auc_score(binary_targets, fraud_probs[1])
            self.log("val_auc", auc, prog_bar=True)
        except ValueError as e:
            log.info(f"There were no true fraud values in the validation set: {e}")

        # Calculate Accuracy
        preds = torch.argmax(fraud_probs, dim=1)  # (b*n/tokens_per_trans)
        accuracy = (preds == binary_targets).float().mean()

        # Lightning automatically computes the average of this loss over all the steps in the epoch
        # The name here becomes a key we can monitor for checkpoint saving
        self.log("val_loss", loss, prog_bar=True) 
        self.log("val_perplexity", perplexity, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

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
        logits = self.model(inputs).logits
        # CrossEntropyLoss was written for classical classification using batch_size x num_classes, so we flatten the token dimension into the batch dimension
        logits = rearrange(logits, 'b n v -> (b n) v')  # (b, n, v) -> (b*n, v)
        targets = rearrange(targets, 'b n -> (b n)')  # (b, n) -> (b*n)
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
            loss_fn = CrossEntropyLoss()
        else:
            raise ValueError(f"Expected 'CrossEntropyLoss' but got {cfg.loss_fn}")

        if cfg.optim == "AdamW":
            optim = AdamW(model.parameters(), lr=cfg.lr)
        else:
            raise ValueError(f"Expected 'AdamW' but got {cfg.optim}")
        
        try:
            lr_scheduler = transformers.get_scheduler(
                name=cfg.lr_scheduler,
                optimizer=optim,
                num_warmup_steps=cfg.warmup_steps,
                num_training_steps=len(train_loader) * cfg.epochs,
            )
        except ValueError as e:
            raise ValueError(f"Expected 'linear' or 'cosine' but got {cfg.lr_scheduler}. {e}") from e
            

        ckpt_dir = os.path.join(get_original_cwd(), cfg.ckpt_path)

        self.wrapped_model = LightingWrapper(model, loss_fn, optim, lr_scheduler, cfg.grad_accum_batches)
        self.ckpt = callbacks.ModelCheckpoint(dirpath=ckpt_dir, monitor="val_perplexity", mode="min")
        self.wandb_logger = L.pytorch.loggers.WandbLogger(project=cfg.project, log_model=cfg.upload_ckpt)
        self.trainer = L.Trainer(
            max_epochs=cfg.epochs,
            val_check_interval=1 / cfg.val_checks_per_epoch,
            accumulate_grad_batches=cfg.grad_accum_batches,
            gradient_clip_val=cfg.gradient_clip_val,
            accelerator="auto",
            callbacks=[self.ckpt],
            logger=self.wandb_logger,
        )


    def train(self):
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