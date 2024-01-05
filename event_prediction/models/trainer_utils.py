import logging

import torch
import transformers
from omegaconf import DictConfig
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from einops import rearrange

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
        # 2. save model checkpoints (see if Jonas has anything in utils.py)
        # 3. Add perplexity metric (see huggingface example)
        # 4. gradient accumulation: https://huggingface.co/learn/nlp-course/chapter7/6#training-with-accelerate
        # 5. Look at the optimizer and hyperparameters used in the hugging face example
        # 6. Add ability to start training from a checkpoint (CV class project)
        # 4. [only if needed ] Use Lighting huggingface Accelerate for multi-gpu (see CV class project)
        for epoch in range(self.epochs):
            self.model.train()
            for step, batch in enumerate(self.train_loader, start=1):
                # inputs: (b, t)
                # targets: (b, t)
                # logits: (b, t, v)
                # b: batch size
                # t: sequence length (tokens)
                # v: vocab size
                self.optim.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                logits = self.model(inputs).logits
                logits = rearrange(logits, 'b t v -> (b t) v')  # (b, n, v) -> (b*n, v)
                targets = rearrange(targets, 'b t -> (b t)')  # (b, n) -> (b*n)
                loss = self.loss_fn(logits, targets)
                loss.backward()
                self.optim.step()
                # if step % 100 == 0:
                log.info(f"Epoch: {epoch} | Step: {step} | Loss: {loss.item()}")

        # self.validate()
        torch.save(self.model.state_dict(), self.chkpt_path)
        return self.chkpt_path

    def validate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        print(f"Validation loss: {avg_loss}")


def get_trainer(
    cfg: DictConfig,
    model: transformers.PreTrainedModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
):
    return Trainer(cfg, model, train_loader, val_loader)
