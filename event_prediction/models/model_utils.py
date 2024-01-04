from transformers import GPT2LMHeadModel, AutoConfig


def get_model(cfg, tokenizer):
    # For convenience use the configuration from a pretrained GPT-2 model, but our actualy GPT-2 model will be trained from scratch
    config = AutoConfig.from_pretrained(
        "gpt2",
        vocab_size=len(tokenizer.vocab),
        n_ctx=cfg.context_length,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)
    return model
