import logging
import os
from typing import Dict, Tuple, Any

from .model_utils import get_model

import torch
import torch.nn.functional as F
import transformers
from einops import rearrange
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.metrics import roc_auc_score
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC

from transformers import DataCollatorForLanguageModeling, AutoTokenizer

log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# metric = BinaryAUROC(thresholds=None)
metric = BinaryAUROC(thresholds=None, compute_on_cpu=True)


class ModelTrainerInterface:
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: AutoTokenizer,
        data_loaders: Dict[str, DataLoader],
        datacollator: DataCollatorForLanguageModeling = None,
        num_cols: int = None,
        label_ids: Dict[str, int] = None,
        row_id: int = -1,
        train_eval: bool = True,
        setup=None
    ):
        self.cfg = cfg

        # DATA
        self.data_loaders = data_loaders
        # self.model = model
        # self.loss_fn = loss_fn
        # self.cols = num_cols
        # self.label_ids = label_ids
        self.setup = {"device": device}
        if setup is not None:
            self.setup = setup

        # Load Model
        model = get_model(
            cfg, tokenizer
        )
        checkpoint = {}
        if cfg.load_checkpoint_path is not None:
            # load from .ckpt
            path = os.path.join(get_original_cwd(), cfg.load_checkpoint_path)
            checkpoint = self.get_checkpoint(path)
            self.model = self.load_model_checkpoint(checkpoint["model_state"])
            log.info(f"Checkpoint loaded from: {path}")
        else:
            # not loaded from checkpoint
            self.model = model

        if train_eval:
            # Optimization params

            if cfg.optim == "AdamW":
                self.optim = AdamW(self.model.parameters(), lr=cfg.lr)
            else:
                log.warning(f"Expected 'AdamW' but got {cfg.optim}, cannot train")
                self.optim = None
                raise NotImplementedError()

            try:
                self.optim.load_state_dict(checkpoint["optim_state"])
            except:
                log.info("Initialized Optimizer from scratch")

            try:
                self.lr_scheduler = transformers.get_scheduler(
                    name=cfg.lr_scheduler,
                    optimizer=self.optim,
                    num_warmup_steps=cfg.warmup_steps,
                    num_training_steps=len(data_loaders.get("train")) * cfg.epochs,
                )

                try:
                    self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
                except:
                    log.info("Initialized Scheduler from scratch")

            except ValueError as e:
                self.lr_scheduler = None
                log.warning(f"Expected 'linear' or 'cosine' but got {cfg.lr_scheduler} so cannot train. {e}")
                raise NotImplementedError()

            except TypeError as e:
                self.lr_scheduler = None
                log.warning(f"Cannot init lr_scheduler {e}")
                raise NotImplementedError()

    def train(self, dataloader) -> Dict[str, Any]:
        self.model.train()
        for data_idx, batch in enumerate(dataloader):
            model_output = self.train_loop(data_idx, batch)

    def train_loop(self, idx, batch):
        model_outputs = {}
        device_batch = self.to_device(batch, keys=["input_ids", "targets", "mask"])
        loss_vals = []
        log_ppls = []
        stream_depth = device_batch["input_ids"].shape[1]

        # TODO do I want to do this? or just run one forward?
        # for seq_idx in range(0, max(1, device_batch["input_ids"].shape[1])): # - self.cfg.train.stream_depth), self.cfg.train.stream_depth):
        #     # Run over seq_dim and dispatch multiple model updates while maintaining state in model_outputs
        #     input_ids = device_batch["input_ids"][:, seq_idx: seq_idx + stream_depth + 1].clone()
        #     # last token is only a target
        #     model_outputs = self.forward(input_ids, **model_outputs)
        #     loss = model_outputs["loss"]
        #
        #     self.backward(loss)
        #     self.optimizer_step()
        #     loss_vals.append(loss.detach())
        #     log_ppls.append(model_outputs["log_perplexity"].detach())
        #
        #     if self.cfg.dryrun:
        #         break
        model_outputs = self.forward(device_batch, **model_outputs)

    def optimizer_step(self):
        raise NotImplementedError()

    def test(self):
        self.model.eval()

        raise NotImplementedError()

    def validate(self):
        self.model.eval()
        pass

    def validation_loop(self):
        pass

    def save_model(self, ckpt_path):
        raise NotImplementedError()

    def get_checkpoint(self, ckpt_path):
        try:
            return torch.load(ckpt_path)
        except:
            log.warning(f"Cannot load checkpoint {ckpt_path}")
            return {}

    def load_model_checkpoint(self, ckpt_path: str, **kwargs: Any,):
        raise NotImplementedError()

    def step(self, batch: dict[str, torch.Tensor]):
        loss = self.forward(**batch)["loss"]
        self.backward(loss)
        self.optimizer_step()
        return loss.detach()

    def to_device(self, batch: dict[str, torch.Tensor], keys: list[str] = ["input_ids"]):
        """Move batch of data into device memory."""
        device_batch = {
            k: v.to(device=self.setup["device"], dtype=torch.long if k == "input_ids" else None, non_blocking=True)
            for k, v in batch.items()
            if k in keys  # Add more keywords here if needed
        }
        return device_batch

    def forward(self, batch, **kwargs):
        inputs = batch["input_ids"]
        targets = batch["targets"]
        mask = batch["mask"]
        model_output = self.model(inputs, targets, mask=mask)  # (b, n, v)
        print(model_output)
        exit()

        # self.accumulated_samples += self.effective_mbs
        # context = self.model.no_sync if self.accumulated_samples < self.current_batch_size else nullcontext
        # with context():
        #     with torch.autocast(**self.amp_settings):
        #         return self.model(*inputs, **kwargs)
        # todo return loss, log_perplexity, logits
        return {"loss": loss, "logits": outputs[:, -1, :], "log_perplexity": loss.clone().detach()}

    @torch.no_grad()
    @torch._dynamo.disable()
    def forward_inference(self, *inputs, **kwargs):
        with torch.autocast(**self.amp_settings):
            outputs = self.model(*inputs, **kwargs)["logits"]
        predictions = outputs.argmax(dim=-1)
        return outputs, predictions

    def backward(self, loss):
        context = self.model.no_sync if self.accumulated_samples < self.current_batch_size else nullcontext
        with context():
            return self.scaler.scale(loss / self.accumulation_steps_expected).backward()


def get_trainer(
    cfg: DictConfig,
    tokenizer: AutoTokenizer,
    dataloaders: Dict[str, DataLoader],
    datacollator: DataCollatorForLanguageModeling = None,
    num_cols: int = None,
    label_ids: Dict[str, int] = None,
    row_id: int = -1,
):
    """
    Return an object that wraps a model, data, and optimization code so we can
    call .train() on it runs and saves checkpoints.
    """
    return ModelTrainerInterface(cfg, tokenizer, data_loaders=dataloaders, datacollator=datacollator, num_cols=num_cols, label_ids=label_ids, row_id=row_id)


# metrics consider moving somewhere:
def get_metrics(wrapper, batch) -> Dict[str, torch.Tensor]:

    # CrossEntropyLoss was written for classical classification using batch_size x num_classes, so we flatten the token dimension into the batch dimension
    inputs_flattened = rearrange(inputs, 'b n -> (b n)')  # (b, n) -> (b*n)
    logits_flattened = rearrange(logits, 'b n v -> (b n) v')  # (b, n, v) -> (b*n, v)
    targets_flattened = rearrange(targets, 'b n -> (b n)')  # (b, n) -> (b*n)
    loss = wrapper.loss_fn(logits_flattened, targets_flattened)
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    auc = 0
    accuracy = 0
    if wrapper.label_ids is not None and wrapper.row_id >= 0:
        # todo flattened?
        auc, accuracy = get_auc_and_perplexity(inputs_flattened, logits_flattened, targets_flattened, wrapper.label_ids, wrapper.row_id)

    return {"loss": loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "auc": auc}


def get_auc_and_perplexity(inputs: torch.Tensor, logits: torch.Tensor, targets: torch.Tensor, label_ids: Dict[str, int], row_id: int):
    # Evaluate generated data as a classification-style problem
    # Here we assume that the last token in each transaction is the one we want to evaluate - the "label".
    # We select the logits and targets for that token position over the context window, then convert the
    # logits into a probability distribution over the label classes. We then calculate the AUC and accuracy.
    # minus 1 because zero-indexed, minus 2 because we want the logits from the token prior to the label token.
    # In our data preparation we align inputs and targets so we don't need to offset them here.
    # That is different from default HF behavior. TODO make it so it doesnt rely on this

    # selected_logits = logits[num_cols - 2:: num_cols]  # (b*n, v) -> (b*n/tokens_per_trans, v)
    # selected_targets = targets[num_cols - 2:: num_cols]  # (b*n) -> (b*n/tokens_per_trans)
    labels = (inputs == row_id).nonzero(as_tuple=True)[0] - 2
    selected_logits = logits[labels]
    selected_targets = targets[labels]

    # Convert logits over all vocabulary to fraud/not-fraud probability for use in accuracy
    not_fraud_id = label_ids["False"]
    is_fraud_id = label_ids["True"]
    notfraud_logits = selected_logits[:, not_fraud_id]  # (b*n/tokens_per_trans, v) -> (b*n/tokens_per_trans)
    isfraud_logits = selected_logits[:, is_fraud_id]
    fraud_logits = torch.stack((notfraud_logits, isfraud_logits))  # (2, b*n/tokens_per_trans)
    fraud_probs = F.softmax(fraud_logits, dim=0)

    # Calculate AUC
    auc = 0
    accuracy = 0
    # try:
    if 1:
        is_fraud_probs = fraud_probs[1]  # (b*n/tokens_per_trans)
        binary_targets = selected_targets == is_fraud_id  # (b*n/tokens_per_trans)
        auc = metric(is_fraud_probs, binary_targets)
        # Calculate Accuracy
        preds = torch.argmax(fraud_probs, dim=0)  # (b*n/tokens_per_trans)
        accuracy = (preds == binary_targets).float().mean()

    # except ValueError as e:
    #     log.info(f"There were no true fraud values in the validation set: {e}")
    #
    return auc, accuracy
