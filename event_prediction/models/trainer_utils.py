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

    def __init__(self, model, loss_fn, optim, lr_scheduler, grad_accum_batches=1, num_cols=None, label_ids=None):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.grad_accum_batches = grad_accum_batches
        self.cols = num_cols
        self.label_ids = label_ids

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
        logits = self.model(inputs).logits # (b, n, v)
        # CrossEntropyLoss was written for classical classification using batch_size x num_classes, so we flatten the token dimension into the batch dimension
        logits = rearrange(logits, 'b n v -> (b n) v')  # (b, n, v) -> (b*n, v)
        targets = rearrange(targets, 'b n -> (b n)')  # (b, n) -> (b*n)
        loss = self.loss_fn(logits, targets)
        try:
            perplexity = torch.exp(loss)
        except OverflowError:
            perplexity = float("inf")

        # Lightning automatically computes the average of this loss over all the steps in the epoch
        # The name here becomes a key we can monitor for checkpoint saving
        self.log("val_loss", loss, prog_bar=True) 
        self.log("val_perplexity", perplexity, prog_bar=True)
        
        # Evaluate generated data as a classification-style problem
        # Here we assume that the last token in each transaction is the one we want to evaluate - the "label".
        # We select the logits and targets for that token position over the context window, then convert the
        # logits into a probability distribution over the label classes. We then calculate the AUC and accuracy.
        if self.cols is not None and self.label_ids is not None:
            # minus 1 because zero-indexed, minus 2 because we want the logits from the token prior to the label token.
            # In our data preparation we align inputs and targets so we don't need to offset them here. That is different from default HF behavior.
            selected_logits = logits[self.cols - 2 : : self.cols]  # (b*n, v) -> (b*n/tokens_per_trans, v)
            selected_targets = targets[self.cols - 2 : : self.cols] # (b*n) -> (b*n/tokens_per_trans)
            
            # Convert logits over all vocabulary to fraud/not-fraud probability for use in accuracy
            not_fraud_id = self.label_ids["False"]
            is_fraud_id = self.label_ids["True"]
            notfraud_logits = selected_logits[:, not_fraud_id] # (b*n/tokens_per_trans, v) -> (b*n/tokens_per_trans)
            isfraud_logits = selected_logits[:, is_fraud_id] 
            fraud_logits = torch.stack((notfraud_logits, isfraud_logits)) # (2, b*n/tokens_per_trans)
            fraud_probs = F.softmax(fraud_logits, dim=0)

            # Calculate AUC
            try:
                binary_targets = selected_targets == is_fraud_id  # (b*n/tokens_per_trans)
                auc = roc_auc_score(binary_targets, fraud_probs[1])
                self.log("val_auc", auc, prog_bar=True)
            except ValueError as e:
                log.info(f"There were no true fraud values in the validation set: {e}")

            # Calculate Accuracy
            preds = torch.argmax(fraud_probs, dim=0)  # (b*n/tokens_per_trans)
            accuracy = (preds == binary_targets).float().mean()
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
        num_cols: int = None,
        label_ids: Tuple[int] = None
    ):
        """
        """
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

        self.wrapped_model = LightingWrapper(model, loss_fn, optim, lr_scheduler, cfg.grad_accum_batches, num_cols, label_ids)
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
        if self.ckpt.best_model_path:
            statedict_path = self.save_statedict(self.ckpt.best_model_path)
        else:
            log.error("No checkpoints successfully saved in training.")
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
    num_cols: int = None,
    label_ids: Tuple[int] = None
):
    """
    Return an object that wraps a model, data, and optimization code so we can
    call .train() on it runs and saves checkpoints.
    """
    return Trainer(cfg, model, train_loader, val_loader, num_cols, label_ids)