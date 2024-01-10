import hydra
import event_prediction
from event_prediction import model_utils, trainer_utils, data_utils
import logging
from typing import Dict

log = logging.getLogger(__name__)


def main_pretrain(cfg, setup=None) -> Dict:
    tokenizer = event_prediction.get_tokenizer(cfg.tokenizer, cfg.data)
    tokenizer.load_vocab_from_file(cfg.data.name, cfg.tokenizer_dir)
    model = model_utils.get_model(
        cfg.model, tokenizer
    )
    model_size = sum(t.numel() for t in model.parameters())
    vocab_size = len(tokenizer.vocab)
    log.info(f"GPT-2 size: {model_size/1000**2:.1f}M parameters")
    log.info(f"Vocab size: {vocab_size}")

    tokenized_string_dataset = data_utils.load_processed_dataset(cfg.data, cfg.processed_data_dir)
    assert isinstance(tokenized_string_dataset, list), f"Expected list of string tokens, instead got {type(tokenized_string_dataset)}"
    log.info(f"Total tokens in dataset: {len(tokenized_string_dataset)}")
    log.info(f"Unique tokens in dataset: {len(set(tokenized_string_dataset))}")
    log.info(f"Num tokens not in vocab: {len(set(tokenized_string_dataset) - tokenizer.vocab)}")

    train_loader, val_loader = data_utils.get_dataloader(cfg.model, tokenizer, tokenized_string_dataset)
    log.info(f"Num train loader batches: {len(train_loader)}")
    log.info(f"Num val loader batches: {len(val_loader)}")
    log.info(f"Dataloader batch size: {train_loader.batch_size}")
    log.info(f"Total tokens in dataloaders: {(len(train_loader) + len(val_loader)) * val_loader.batch_size}")

    trainer = trainer_utils.get_trainer(cfg.model, model, train_loader, val_loader)
    weights_filepath = trainer.train()
    
    metrics = {}
    return metrics



@hydra.main(
    config_path="event_prediction/config",
    config_name="pretrain",
    version_base="1.3",
)
def launch(cfg):
    event_prediction.utils.main_launcher(cfg, main_pretrain, job_name="pretrain")


if __name__ == "__main__":
    launch()
