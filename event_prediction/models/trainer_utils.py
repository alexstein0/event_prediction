import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling


def get_trainer(cfg, model, tokenizer, tokenized_dataset):
    # https://huggingface.co/learn/nlp-course/en/chapter7/6?fw=pt#initializing-a-new-model

    # tokenizer.pad_token = tokenizer.eos_token
    # The data collator seperates the dataset into batches, and pads sequences that are
    # shorter than the context window. It also creates self-supervised labels in either
    # a causal or masked language modeling fashion.
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # Use mixed-precision training if on GPU
    cfg.train_args.fp16 = torch.cuda.is_available() and hasattr(torch.cuda, 'amp')
    args = TrainingArguments(**cfg.train_args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['train'],
    )

    return trainer