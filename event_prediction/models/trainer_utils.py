import logging
import os
from typing import Dict, Tuple, Any

from .model_utils import get_model

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
# from torchmetrics.functional.classification import binary_auroc
from torchmetrics.classification import BinaryAUROC

from transformers import DataCollatorForLanguageModeling, AutoTokenizer

log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

metric = BinaryAUROC(thresholds=None)


class LightingWrapper(L.LightningModule):
    """
    Wraps a torch.nn.Module or compatible huggingface model in a LightningModule class for automated training.
    A LightningModule tightly couples the model with an optimizer and train step specifications. This allows
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

    def __init__(self, model, loss_fn=None, optim=None, lr_scheduler=None, grad_accum_batches=1,
                 num_cols: int = None, label_ids: Dict[str, int] = None):
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
        metric_dict = get_metrics(self, batch)

        loss = metric_dict["loss"]
        perplexity = metric_dict["perplexity"]
        auc = metric_dict["auc"]
        accuracy = metric_dict["accuracy"]

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_perplexity", perplexity, prog_bar=True)
        self.log("train_auc", auc, prog_bar=True)
        self.log("train_accuracy", accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Required LightningModule method.
        Gets called from .fit() and automatically handles .eval() and calls mentioned in training_step()"""
        metric_dict = get_metrics(self, batch)

        loss = metric_dict["loss"]
        perplexity = metric_dict["perplexity"]
        auc = metric_dict["auc"]
        accuracy = metric_dict["accuracy"]

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_perplexity", perplexity, prog_bar=True)
        self.log("val_auc", auc, prog_bar=True)
        self.log("val_accuracy", accuracy, prog_bar=True)

        return loss

    # def get_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    #     """
    #     Helper function for computing loss.
    #
    #     batch = (inputs, targets)
    #     inputs shape: (b, t)
    #     targets shape: (b, t)
    #     logits shape: (b, t, v)
    #     b: batch size
    #     t: sequence length (tokens)
    #     v: vocab size
    #     """
    #     inputs, targets = batch
    #     logits = self.model(inputs).logits
    #     # CrossEntropyLoss was written for classical classification using batch_size x num_classes, so we flatten the token dimension into the batch dimension
    #     logits = rearrange(logits, 'b n v -> (b n) v')  # (b, n, v) -> (b*n, v)
    #     targets = rearrange(targets, 'b n -> (b n)')  # (b, n) -> (b*n)
    #     loss = self.loss_fn(logits, targets)
    #     return loss
    
    @staticmethod
    def extract_statedict(ckpt_path: str) -> Dict:
        """Helper function for extracting the statedict from a LightningModule checkpoint."""
        checkpoint = torch.load(ckpt_path)
        lightning_statedict = checkpoint["state_dict"]
        statedict = {k.replace("model.", ""): v for k, v in lightning_statedict.items()}
        return statedict


class ModelTrainerInterface:
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: AutoTokenizer,
        data_loaders: Dict[str, DataLoader],
        datacollator: DataCollatorForLanguageModeling = None,
        num_cols: int = None,
        label_ids: Dict[str, int] = None
    ):
        """
        """
        # DATA
        self.data_loaders = data_loaders
        # self.model = model
        # self.loss_fn = loss_fn
        # self.cols = num_cols
        # self.label_ids = label_ids

        # Lightning things
        ckpt_dir = os.path.join(get_original_cwd(), cfg.ckpt_path)
        self.wandb_logger = L.pytorch.loggers.WandbLogger(project=cfg.project, log_model=cfg.upload_ckpt)
        self.ckpt = callbacks.ModelCheckpoint(dirpath=ckpt_dir, monitor="val_perplexity", mode="min")

        # Load Model

        # load default HF model
        model = get_model(
            cfg, tokenizer
        )

        # Optimization params
        if cfg.loss_fn == "CrossEntropyLoss":
            loss_fn = CrossEntropyLoss()
        else:
            log.warning(f"Expected 'CrossEntropyLoss' but got {cfg.loss_fn}, cannot train")
            loss_fn = None

        if cfg.optim == "AdamW":
            optim = AdamW(model.parameters(), lr=cfg.lr)
        else:
            log.warning(f"Expected 'AdamW' but got {cfg.optim}, cannot train")
            optim = None

        try:
            lr_scheduler = transformers.get_scheduler(
                name=cfg.lr_scheduler,
                optimizer=optim,
                num_warmup_steps=cfg.warmup_steps,
                num_training_steps=len(data_loaders.get("train")) * cfg.epochs,
            )
        except ValueError as e:
            log.warning(f"Expected 'linear' or 'cosine' but got {cfg.lr_scheduler} so cannot train. {e}")
            lr_scheduler = None

        # if pretrained, add state dict
        if cfg.state_dict_path is not None:
            # load from .pth
            path = os.path.join(get_original_cwd(), cfg.state_dict_path)
            state_dict = torch.load(path, device)
            model.load_state_dict(state_dict)
            model.eval()
            self.wrapped_model = LightingWrapper(model, loss_fn, optim, lr_scheduler, cfg.grad_accum_batches, num_cols, label_ids)
            log.info(f"Loaded model from state_dict: {cfg.state_dict_path}")
        elif cfg.load_checkpoint_path is not None:
            # load from .ckpt
            path = os.path.join(get_original_cwd(), cfg.load_checkpoint_path)
            self.wrapped_model = self.load_checkpoint(path, num_cols=num_cols, label_ids=label_ids)
            log.info(f"Not using pth, will use ckpt within Lightning")
        else:
            # not loaded from checkpoint
            self.wrapped_model = LightingWrapper(model, loss_fn, optim, lr_scheduler, cfg.grad_accum_batches, num_cols, label_ids)

        # todo is this going to work for test only?
        self.trainer = L.Trainer(
            max_epochs=cfg.epochs,
            val_check_interval=1 / cfg.val_checks_per_epoch,
            accumulate_grad_batches=cfg.grad_accum_batches,
            gradient_clip_val=cfg.gradient_clip_val,
            accelerator="auto",
            callbacks=[self.ckpt],
            logger=self.wandb_logger,
            log_every_n_steps=1  # TODO change this 50 is default
        )

    def train(self) -> Dict[str, Any]:
        self.trainer.fit(self.wrapped_model, self.data_loaders.get("train"), self.data_loaders.get("val"))
        self.wandb_logger.experiment.finish()
        output_dict = {}
        if self.ckpt.best_model_path:
            statedict_path = self.save_statedict(self.ckpt.best_model_path)
            output_dict["statedict_path"] = statedict_path
        else:
            log.error("No checkpoints successfully created during training.")

        # output_dict["loss"] = self.loss
        return output_dict

    def test(self):
        self.trainer.test(self.wrapped_model, self.data_loaders.get("test"))
        self.wandb_logger.experiment.finish()

    def save_statedict(self, ckpt_path):
        """Extract and save the statedict from a PyTorch Lightning checkpoint."""        
        statedict = LightingWrapper.extract_statedict(ckpt_path)
        statedict_path = ckpt_path.replace(".ckpt", ".pth")
        torch.save(statedict, statedict_path)
        return statedict_path

    def load_checkpoint(self, ckpt_path: str, **kwargs: Any,):
        wrapper = LightingWrapper.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            kwargs=kwargs
            # hparams_file="/path/to/test_tube/experiment/version/hparams.yaml",
            # map_location=None
        )
        return wrapper


    def get_model(self):
        return self.wrapped_model


def get_trainer(
    cfg: DictConfig,
    tokenizer: AutoTokenizer,
    dataloaders: Dict[str, DataLoader],
    datacollator: DataCollatorForLanguageModeling = None,
    num_cols: int = None,
    label_ids: Dict[str, int] = None
):
    """
    Return an object that wraps a model, data, and optimization code so we can
    call .train() on it runs and saves checkpoints.
    """
    return ModelTrainerInterface(cfg, tokenizer, data_loaders=dataloaders, datacollator=datacollator, num_cols=num_cols, label_ids=label_ids)


#
# class Evaluator:
#     def __init__(
#         self,
#         cfg: DictConfig,
#         model: transformers.PreTrainedModel = None,
#         data_loader: DataLoader = None,
#         num_cols: int = None,
#         label_ids: Dict[str, int] = None
#     ):
#         """
#         """
#         self.data_loader = data_loader
#
#         if cfg.loss_fn == "CrossEntropyLoss":
#             loss_fn = CrossEntropyLoss()
#         else:
#             raise ValueError(f"Expected 'CrossEntropyLoss' but got {cfg.loss_fn}")
#         self.model = model
#         self.loss_fn = loss_fn
#         self.cols = num_cols
#         self.label_ids = label_ids
#
#         # self.wrapped_model = LightingWrapper(model, loss_fn, num_cols=num_cols, label_ids=label_ids)
#
#     def eval_loop(self, batch) -> float:
#         """This is a hack because evaluator doesn't need lightning wrapper (or at least idk how to make it work)"""
#         metric_dict = get_metrics(self, batch)
#         loss = metric_dict.get("loss", 0.0)
#         perplexity = metric_dict.get("perplexity", None)
#         auc = metric_dict.get("auc", None)
#         accuracy = metric_dict.get("accuracy", None)
#
#         print("train_loss", loss.item())
#         print("train_perplexity", perplexity.item())
#         print("train_auc", auc.item())
#         print("train_accuracy", accuracy.item())
#
#         return loss
#
#     def evaluate(self):
#         losses = []
#         for idx, batch in enumerate(self.data_loader):
#             print(f"batch id: {idx}")
#             loss = self.eval_loop(batch)
#             losses.append(loss)
#         return sum(losses) / len(losses)


# def get_eval(
#     cfg: DictConfig,
#     model: transformers.PreTrainedModel = None,
#     data_loader: DataLoader = None,
#     # datacollator: DataCollatorForLanguageModeling = None,
#     num_cols: int = None,
#     label_ids: Dict[str, int] = None
# ):
#     """
#     Return an object that wraps a model, data, and optimization code, so we can
#     call .train() on it runs and saves checkpoints.
#     """
#     return Evaluator(cfg, model, data_loader, num_cols, label_ids)


# metrics consider moving somewhere:
def get_metrics(wrapper, batch) -> Dict[str, torch.Tensor]:
    inputs, targets = batch
    logits = wrapper.model(inputs).logits  # (b, n, v)

    # CrossEntropyLoss was written for classical classification using batch_size x num_classes, so we flatten the token dimension into the batch dimension
    logits_flattened = rearrange(logits, 'b n v -> (b n) v')  # (b, n, v) -> (b*n, v)
    targets_flattened = rearrange(targets, 'b n -> (b n)')  # (b, n) -> (b*n)
    loss = wrapper.loss_fn(logits_flattened, targets_flattened)
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    auc = 0
    accuracy = 0
    if wrapper.cols is not None and wrapper.label_ids is not None:
        # todo flattened?
        auc, accuracy = get_auc_and_perplexity(logits_flattened, targets_flattened, wrapper.cols, wrapper.label_ids)

    return {"loss": loss,
            "perplexity": perplexity,
            "accuracy": accuracy,
            "auc": auc}


def get_auc_and_perplexity(logits: torch.Tensor, targets: torch.Tensor, num_cols: int, label_ids: Dict[str, int]):
    # Evaluate generated data as a classification-style problem
    # Here we assume that the last token in each transaction is the one we want to evaluate - the "label".
    # We select the logits and targets for that token position over the context window, then convert the
    # logits into a probability distribution over the label classes. We then calculate the AUC and accuracy.
    # minus 1 because zero-indexed, minus 2 because we want the logits from the token prior to the label token.
    # In our data preparation we align inputs and targets so we don't need to offset them here.
    # That is different from default HF behavior. TODO make it so it doesnt rely on this
    selected_logits = logits[num_cols - 2:: num_cols]  # (b*n, v) -> (b*n/tokens_per_trans, v)
    selected_targets = targets[num_cols - 2:: num_cols]  # (b*n) -> (b*n/tokens_per_trans)

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
