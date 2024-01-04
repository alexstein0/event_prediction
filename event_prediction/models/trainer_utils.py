from transformers import Trainer, TrainingArguments


def get_trainer(cfg, model, tokenizer, tokenized_datasets, data_collator):
    # https://huggingface.co/learn/nlp-course/en/chapter7/6?fw=pt#initializing-a-new-model

    args = TrainingArguments(cfg.training_args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"],
    )

    return trainer