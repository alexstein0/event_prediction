import torch.nn as nn
import torch
from transformers import GPT2LMHeadModel, AutoConfig, GPT2Model, AutoModelForCausalLM
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from einops import rearrange

import logging

from event_prediction import tokenizer_utils
from torchmetrics.classification import BinaryAUROC


metric = BinaryAUROC(thresholds=None, compute_on_cpu=True)

log = logging.getLogger(__name__)


# ideas adapted from fata-trans
class Decoder(nn.Module):
    """Decoder model (can be used with GPT pretrained)"""
    def __init__(self, cfg, tokenizer):
        super().__init__()

        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer.vocab),
            n_ctx=cfg.context_length,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        if cfg.loss_fn == "CrossEntropyLoss":
            self.loss_fn = CrossEntropyLoss()
        else:
            log.warning(f"Expected 'CrossEntropyLoss' but got {cfg.loss_fn}, cannot train")
            raise NotImplementedError()

        self.model = GPT2LMHeadModel(config)
        classification_info = tokenizer_utils.get_classification_options(tokenizer, label_in_last_col=True) #, label_col_prefix=label_col_id)
        self.num_cols = classification_info["num_cols"]
        self.label_ids = classification_info["label_ids"]
        self.tokenizer = tokenizer

    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor, *args, **kwargs):
        mask = (input_ids != 0).int()  # todo mask
        if "mask" in kwargs:
            mask = kwargs["mask"]
        outputs = self.model(input_ids)

        logits = outputs["logits"]
        #todo try masking every non label col

        inputs_flattened = rearrange(input_ids, 'b n -> (b n)')  # (b, n) -> (b*n)
        logits_flattened = rearrange(logits, 'b n v -> (b n) v')  # (b, n, v) -> (b*n, v)
        targets_flattened = rearrange(targets, 'b n -> (b n)')  # (b, n) -> (b*n)
        mask_flattened = rearrange(mask, 'b n -> (b n)')

        selected_logits = logits_flattened[mask_flattened == 1, :]
        selected_targets = targets_flattened[mask_flattened == 1]

        loss = self.loss_fn(logits_flattened, targets_flattened)

        # Convert logits over all vocabulary to fraud/not-fraud probability for use in accuracy
        not_fraud_id = self.label_ids["False"]
        is_fraud_id = self.label_ids["True"]
        notfraud_logits = selected_logits[:, not_fraud_id]  # (b*n/tokens_per_trans, v) -> (b*n/tokens_per_trans)
        isfraud_logits = selected_logits[:, is_fraud_id]
        fraud_logits = torch.stack((notfraud_logits, isfraud_logits))  # (2, b*n/tokens_per_trans)
        fraud_probs = F.softmax(fraud_logits, dim=0)

        # Calculate AUC
        is_fraud_probs = fraud_probs[1]  # (b*n/tokens_per_trans)
        binary_targets = selected_targets == is_fraud_id  # (b*n/tokens_per_trans)
        auc = metric(is_fraud_probs, binary_targets)
        # Calculate Accuracy
        preds = torch.argmax(fraud_probs, dim=0)  # (b*n/tokens_per_trans)
        accuracy = (preds == binary_targets).float().mean()

        return {"loss": loss, "logits": outputs["logits"][:, -1, :], "log_perplexity": loss.clone().detach(), "accuracy": accuracy, "auc": auc}


class RowEncoder(nn.Module):
    """This is my user defined encoder to move a row into embedding space before doing autoregressive step"""
    def __init__(self, n_cols: int, vocab_size: int, hidden_size: int, col_hidden_size: int,
                 nheads: int = 8, nlayers: int = 1):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size, col_hidden_size)

        enc = nn.TransformerEncoderLayer(d_model=col_hidden_size, nhead=nheads, dim_feedforward=col_hidden_size)
        self.encoder = nn.TransformerEncoder(enc, num_layers=nlayers)

        self.linear = nn.Linear(col_hidden_size * n_cols, hidden_size)
        # self.hidden_size = hidden_size
        # self.col_hidden_size = col_hidden_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embeddings(input_ids)
        embeds_shape = list(embedded.size())

        embedded = embedded.view([-1] + embeds_shape[-2:])
        embedded = embedded.permute(1, 0, 2)
        embedded = self.encoder(embedded)
        embedded = embedded.permute(1, 0, 2)
        embedded = embedded.contiguous().view(embeds_shape[0:2] + [-1])

        embedded = self.linear(embedded)

        return embedded


class HierarchicalModel(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids: torch):
        return self.decoder(self.encoder(input_ids))
