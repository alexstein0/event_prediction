from transformers import GPT2LMHeadModel, AutoConfig, GPT2Model, AutoModelForCausalLM
from .EPmodels import RowEncoder, HierarchicalModel
import torch
from hydra.utils import get_original_cwd
import os


def get_model(model_cfg, tokenizer):
    if model_cfg.model_type == "encoder":
        model = get_embedding_model(model_cfg, tokenizer)

    elif model_cfg.model_type == "decoder":
        model = get_decoder_model(model_cfg, tokenizer)
    else:
        raise ValueError(f"NO MODEL TYPE: {model_cfg.model_type}")

    # todo add small decoder at end
    # if model_cfg.has_small_decoder:
    #     # small head
    #     raise NotImplementedError()

    # todo hierarchical
    # if encoder is not None and decoder is not None:
    #     return HierarchicalModel(encoder, decoder)
    # elif encoder is not None:
    #     return encoder
    # elif decoder is not None:
    #     return decoder
    # else:
    #     return None
    return model


def get_decoder_model(cfg, tokenizer):
    # For convenience use the configuration from a pretrained GPT-2 model, but our actualy GPT-2 model will be trained from scratch
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer.vocab),
        n_ctx=cfg.context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    # model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    return model


def get_embedding_model(cfg, tokenizer):
    model = RowEncoder(n_cols=cfg.num_encoded_columns,
                       vocab_size=len(tokenizer.get_vocab()),
                       hidden_size=cfg.hidden_size,
                       col_hidden_size=cfg.col_hidden_size,
                       nheads=cfg.num_heads,
                       nlayers=cfg.num_layers
                       )
    return model