import logging
import os
import datetime
from typing import Dict, Tuple, Any, List

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
from torchmetrics.functional.classification import binary_auroc, multiclass_auroc, f1_score

from event_prediction import tokenizer_utils, utils, get_data_processor, data_preparation

import time
import json

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback


from transformers import DataCollatorForLanguageModeling, AutoTokenizer


log = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelTrainerInterface:
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer: AutoTokenizer,
        data_loaders: Dict[str, DataLoader],
        train_eval: bool = True,
        setup=None
    ):

        # data processing configs
        data_processor = get_data_processor(cfg.data)
        classification_info = tokenizer_utils.get_classification_options(tokenizer, label_in_last_col=True)
        self.num_cols = classification_info["num_cols"]
        self.label_ids = classification_info["label_ids"]
        self.tokenizer = tokenizer

        _, self.col_to_loc = data_preparation.get_col_to_id_dict(data_processor.get_data_cols(), dataset=None)
        consolidation_map = cfg.data.consolidate_columns
        self.consolidation_map = {}
        if consolidation_map is not None and cfg.consolidate:
            assert len(consolidation_map) == 1, "only works for target now"
            for col_name, col_mapping in consolidation_map.items():
                assert len(col_mapping) == len(self.label_ids), "consolidation maps must have the same number of columns"
                for label_value, label_id in col_mapping.items():
                    self.consolidation_map[self.label_ids[label_value]] = label_id

        # DATA
        self.train_loader = data_loaders.get("train", None)
        if self.train_loader is None:
            log.info("No training data loaded!")

        self.valid_loader = data_loaders.get("test", None)
        if self.valid_loader is None:
            log.info("No test data loaded!")

        self.epochs = cfg.model.epochs
        self.model_context_length = self.num_cols*cfg.model.seq_length
        self.randomize_order = cfg.model.randomize_order
        self.metric_calc_mode = cfg.model.metric_calc_mode

        if cfg.model.track_preds:
            # setting to none will track everything possible
            self.tracked_model_output_training = None
            self.tracked_model_output_eval = None
        else:
            self.tracked_model_output_training = ["loss", "loss_sequence", "accuracy"]
            self.tracked_model_output_eval = ["loss", "loss_sequence", "accuracy", "target_inds", "target_probs"]

        # todo is there a cleaner way of doing these?
        cfg.model.context_length = self.model_context_length
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        if cfg.model_save_name is not None:
            self.model_save_name = cfg.model_save_name
        else:
            self.model_save_name = f"{cfg.name}_{timestamp}"
        self.checkpoint_save_name = cfg.checkpoint_save_name
        log.info(f"Save name: {self.model_save_name}")

        self.setup = {
            "device": device,
            "wandb_enabled": cfg.wandb.enabled,
        }
        if setup is not None:
            self.setup = setup

        # Load Model
        model = get_model(
            cfg.model, tokenizer
        )
        checkpoint = {}
        if cfg.model_save_name is not None:
            # load from .ckpt
            path = os.path.join(get_original_cwd(), cfg.model_dir, cfg.model_save_name, cfg.checkpoint_name)
            checkpoint = self.get_checkpoint(path)
            self.model = self.load_model_checkpoint(model, checkpoint)
            # self.model = checkpoint
            log.info(f"Checkpoint loaded from: {path}")
        else:
            # not loaded from checkpoint
            log.info(f"No checkpoint loaded")
            self.model = model

        self.model.to(device)

        if train_eval:
            # Optimization params
            if cfg.model.optim == "AdamW":
                lr = cfg.model.lr
                self.optim = AdamW(self.model.parameters(), lr=lr)
            else:
                log.warning(f"Expected 'AdamW' but got {cfg.model.optim}, cannot train")
                self.optim = None
                raise NotImplementedError()

            try:
                self.optim.load_state_dict(checkpoint["optim_state"])
            except:
                log.info("Initialized Optimizer from scratch")

            # # TODO scheduler doesnt do anything yet
            # try:
            #     self.lr_scheduler = transformers.get_scheduler(
            #         name=cfg.model.lr_scheduler,
            #         optimizer=self.optim,
            #         num_warmup_steps=cfg.model.warmup_steps,
            #         num_training_steps=len(data_loaders.get("train")) * cfg.model.epochs,
            #     )
            #
            #     try:
            #         self.lr_scheduler.load_state_dict(checkpoint["scheduler_state"])
            #     except:
            #         log.info("Initialized Scheduler from scratch")
            #
            # except ValueError as e:
            #     self.lr_scheduler = None
            #     log.warning(f"Expected 'linear' or 'cosine' but got {cfg.model.lr_scheduler} so cannot train. {e}")
            #     raise NotImplementedError()
            #
            # except TypeError as e:
            #     self.lr_scheduler = None
            #     log.warning(f"Cannot init lr_scheduler {e}")
            #     raise NotImplementedError()

        self.steps = 0
        self.epoch = 0
        self.cfg = cfg

    def train(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        log.info(f"Running training for {self.epochs} epochs")
        stats = dict()
        validation_stats = dict()
        training_stats = dict()
        start_time = time.time()
        last_print_time = start_time
        eval_time = 0
        last_log_step = -1

        for epoch in range(self.epochs):
            self.epoch = epoch
            self.train_loader.dataset.set_epoch = epoch
            self.valid_loader.dataset.set_epoch = epoch
            epoch_start_time = time.time()
            epoch_start_step = self.steps
            for data_idx, batch in enumerate(self.train_loader):
                self.optim.zero_grad()

                model_output = self.train_loop(data_idx, batch)
                stats = self.add_model_output_stats(model_output, stats, self.tracked_model_output_training)
                if (self.steps + 1) % self.cfg.impl.print_loss_every_nth_step == 0:
                    training_stats = self.gather_stats(stats)
                    elapsed_times = utils.get_time_deltas(start_time, last_print_time, set_format=False)

                    training_stats["total_elapsed_time"] = elapsed_times[0]
                    training_stats["step_avg_time"] = (elapsed_times[1] - eval_time) / (self.steps - last_log_step)
                    # note that average time subtracts eval time but not saved time
                    self.log(f"Train Step: {self.steps + 1}", training_stats, is_training=True)

                    last_log_step = self.steps
                    last_print_time = time.time()
                    eval_time = 0

                if (self.steps + 1) % self.cfg.impl.run_eval_every_nth_step == 0:
                    # Todo change the param, change the data
                    eval_start_time = time.time()
                    validation_stats = self.validate()
                    elapsed_times = utils.get_time_deltas(start_time, eval_start_time, set_format=False)
                    validation_stats["total_elapsed_time"] = elapsed_times[0]
                    eval_time += elapsed_times[1]  # time elapsed for eval (subtracted from training time)
                    validation_stats["eval_time"] = eval_time
                    self.log(f"Eval step: {self.steps + 1}", validation_stats, is_training=False)

                if (self.cfg.impl.save_every_nth_step > 0 and
                        (self.steps + 1) % self.cfg.impl.save_every_nth_step == 0 and
                        self.steps != 0 and self.cfg.impl.save_intermediate_checkpoints):
                    if self.checkpoint_save_name is not None:
                        name = self.checkpoint_save_name
                    else:
                        name = f"checkpoint_step_{self.steps + 1}"
                    model_path = self.save_model(name)
                    log.info(f"Saving to {model_path}")

                self.steps += 1

                # todo
                # dont calculate all training stats
                # learning rate tests for training

            training_stats, stats = self.reset_intermediate_stats(stats)
            eval_start_time = time.time()
            validation_stats = self.validate()
            elapsed_times = utils.get_time_deltas(start_time, eval_start_time, epoch_start_time, set_format=False)
            total_time = elapsed_times[0]
            eval_time += elapsed_times[1]
            epoch_time = elapsed_times[2]
            training_stats["step_avg_time"] = (elapsed_times[2] - eval_time) / (self.steps - epoch_start_step)
            eval_time = 0

            validation_stats["total_elapsed_time"] = total_time
            validation_stats["epoch_time"] = epoch_time
            self.log(f"Training for epoch: {epoch}", training_stats, is_training=True)
            self.log(f"Eval for epoch: {epoch}", validation_stats, is_training=False)
            if self.cfg.impl.save_intermediate_checkpoints:
                name = f"checkpoint_end_{epoch}"
                model_path = self.save_model(name)
                log.info(f"Saving to {model_path}")

        # todo add eval every once in a while
        # log to wandb
        # checkpoint model
        # track number of steps and ensure that scheduling is working correctly
        return validation_stats, training_stats

    def train_loop(self, idx, batch):
        self.model.train()
        # batch["input_ids"][:] = 4
        # todo loop over devices/minibatches

        if "mask" not in batch:
            batch["mask"] = (batch["input_ids"] != 0).int()

        device_batch = self.to_device(batch, keys=["input_ids", "mask", "labels"])
        # loss_vals = []
        # log_ppls = []
        # stream_depth = device_batch["input_ids"].shape[1]
        # model_outputs = self.forward(device_batch, **model_outputs)

        # input_ids = device_batch["input_ids"]
        # targets = device_batch["targets"]

        outputs = self.forward(device_batch)
        loss = outputs["loss"]
        self.backward(loss)
        self.optimizer_step()
        metrics = self.calc_metrics(device_batch, outputs)
        return metrics

    def optimizer_step(self):
        self.optim.step()

    def test(self):
        stats = self.validate()  # todo, is this different than validation?
        self.log(f"EVALUATION", stats, is_training=False)
        return stats

    def validate(self):
        self.model.eval()
        stats = {}
        for data_idx, batch in enumerate(self.valid_loader):
            model_output = self.validation_loop(data_idx, batch)
            stats = self.add_model_output_stats(model_output, stats, self.tracked_model_output_eval)
        calced_stats = self.gather_stats(stats)
        return calced_stats

    def validation_loop(self, idx, batch):
        device_batch = self.to_device(batch, keys=["input_ids", "mask", "labels"])
        outputs = self.forward_inference(device_batch)
        metrics = self.calc_metrics(device_batch, outputs)
        return metrics

    def save_model(self, name: str):
        model_path = os.path.join(get_original_cwd(), self.cfg.model_dir, self.model_save_name)
        full_path = os.path.join(model_path, name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.model.state_dict(), full_path)
        train_ids = self.train_loader.dataset.user_ids
        val_ids = self.valid_loader.dataset.user_ids
        train_test_split = {"train": train_ids, "test": val_ids}
        with open(os.path.join(model_path, "train_test_split.json"), "w") as f:
            json.dump(train_test_split, f)

        return full_path

    def get_checkpoint(self, ckpt_path):
        try:
            return torch.load(ckpt_path)
        except:
            log.warning(f"Cannot load checkpoint {ckpt_path}")
            return {}

    def add_model_output_stats(self, model_outputs: Dict[str, Any],
                               intermediate_stats: Dict[str, Any],
                               stats_list: List[str] = None,  # can pass in list of desired stats
                               ) -> Dict[str, Any]:
        intermediate_stats["batches"] = intermediate_stats.get("batches", 0) + 1
        if stats_list is None:
            stats_list = model_outputs.keys()
        for k, v in model_outputs.items():
            if k not in stats_list:
                continue
            prev = intermediate_stats.get(k, [])
            if len(v.shape) == 0:
                prev.append(v.detach().item())
            else:
                prev.append(v)
            intermediate_stats[k] = prev
        return intermediate_stats

    def reset_intermediate_stats(self, stats: Dict[str, Any]) -> (Dict[str, Any], Dict[str, Any]):
        output_stats = self.gather_stats(stats)
        stats = {}
        return output_stats, stats

    def gather_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        output_stats = {}
        count = stats.get("batches", 1)
        output_stats["batches"] = count

        if "loss" in stats:
            output_stats["loss"] = sum(stats["loss"]) / count
        if "accuracy" in stats:
            output_stats["accuracy"] = sum(stats["accuracy"]) / count
        if "target_inds" in stats:
            target_inds = torch.cat(stats["target_inds"])
        if "target_probs" in stats:
            target_probs = torch.cat(stats["target_probs"])

        # Calculate AUC
        try:
            auc = multiclass_auroc(target_probs.T, target_inds, num_classes=target_probs.shape[0], thresholds=None)
            output_stats["auc"] = auc.item()
            output_stats["f1"] = f1_score(target_probs.T, target_inds, num_classes=target_probs.shape[0], task="multiclass")
        except:
            pass

        if "accuracy" in stats:
            output_stats["accuracy"] = sum(stats["accuracy"]) / count

        output_stats = utils.collect_memory_usage(output_stats, device)
        return output_stats

    def calc_metrics(self, device_batch: Dict[str, Any], model_outputs: Dict[str, Any]) -> Dict[str, Any]:
        metrics = {}
        if "loss" in model_outputs:
            loss = model_outputs["loss"]
            metrics["loss"] = loss.clone().detach().to("cpu")
        logits = model_outputs["logits"]

        # Convert logits over all vocabulary to target probability for use in accuracy
        label_mask = device_batch["mask"][:, 1:]  # mask of where the labels are

        # # where should metrics be calculated (like auc only on row labels)
        if self.metric_calc_mode == "last":
            mask = torch.zeros_like(label_mask, device=label_mask.device)
            mask[torch.arange(label_mask.size(0)), torch.argmax((label_mask == 1).long().cumsum(dim=1) * label_mask, dim=1)] = 1
        elif self.metric_calc_mode == "labels":
            mask = label_mask
        else:
            raise

        mask_flattened_shifted = mask.reshape(-1)
        # todo bos eos
        shifted_outputs = logits[..., :-1, :].contiguous()
        logits_flattened = shifted_outputs.view(-1, shifted_outputs.shape[-1])
        shifted_labels = device_batch["labels"].contiguous()
        labels_flattened = shifted_labels.view(-1)

        selected_logits = logits_flattened[mask_flattened_shifted == 1, :]
        selected_targets = labels_flattened[mask_flattened_shifted == 1]

        target_logits = []
        target_inds = torch.zeros_like(selected_targets) - 1
        consolidation_map = []
        for i, (word, word_id) in enumerate(self.label_ids.items()):
            target_logits.append(selected_logits[:, word_id])
            target_inds[selected_targets == word_id] = i
            consolidation_map.append(self.consolidation_map.get(word_id, -1))
        target_logits = torch.stack(target_logits)
        consolidation_map = torch.tensor(consolidation_map).to(target_inds.device)
        target_probs = F.softmax(target_logits, dim=0)

        if len(self.consolidation_map) > 1:
            target_probs, target_inds = self.consolidate_column_values(consolidation_map, target_probs, target_inds)
        assert (target_inds >= 0).all(), "all possible logits found"
        assert target_probs.sum(dim=0).allclose(torch.tensor(1.0)), "all probs add to 1"

        # Calculate AUC
        metrics["target_probs"] = target_probs.to("cpu")
        metrics["target_inds"] = target_inds.to("cpu")

        preds = torch.argmax(target_probs, dim=0)  # (b*n/tokens_per_trans)
        accuracy = (preds == target_inds).float().mean()
        metrics["accuracy"] = accuracy.detach().to("cpu")

        return metrics

    def consolidate_column_values(self, consolidation_map, logits, inds):
        classes = consolidation_map.unique()
        output_logits = []
        for c in classes:
            output_logits.append(logits[consolidation_map == c].sum(dim=0))

        output_logits = torch.stack(output_logits)
        inds = consolidation_map[inds]
        return output_logits, inds

    def load_model_checkpoint(self, model, checkpoint: str, **kwargs: Any):
        model.load_state_dict(checkpoint)
        return model

    # def step(self, batch: dict[str, torch.Tensor]):
    #     loss = self.forward(**batch)["loss"]
    #     self.backward(loss)
    #     self.optimizer_step()
    #     return loss.detach()

    def to_device(self, batch: dict[str, torch.Tensor], keys: list[str] = ["input_ids"]):
        """Move batch of data into device memory."""
        device_batch = {}
        for k, v in batch.items():
            if k in keys:
                v = v.to(device=self.setup["device"], dtype=torch.long if k == "input_ids" else None, non_blocking=True)
            device_batch[k] = v
        return device_batch

    def forward(self, batch, **kwargs):
        input_ids = batch["input_ids"].clone()
        labels = batch["labels"].clone()
        mask = batch["mask"].clone()
        model_outputs = self.model(input_ids=input_ids, target_mask=mask, labels=labels)
        return model_outputs

    @torch.no_grad()
    @torch._dynamo.disable()
    def forward_inference(self, batch, **kwargs):
        input_ids = batch["input_ids"].clone()
        mask = batch["mask"].clone()
        model_outputs = self.model(input_ids=input_ids, target_mask=mask)
        return model_outputs


    def backward(self, loss):
        loss.backward()
        # todo microbatching
        # context = self.model.no_sync if self.accumulated_samples < self.current_batch_size else nullcontext
        # with context():
        #     return self.scaler.scale(loss / self.accumulation_steps_expected).backward()

    def log(self, message: str, metrics: Dict[str, Any], is_training: bool):
        kw_str = ""
        kw_str += f'''{f"batches: {metrics['batches']}":14s} ''' if 'batches' in metrics else ""
        kw_str += f'''{f"Total dur: {utils.format_time(metrics['total_elapsed_time'], 2)}":25s} ''' if 'total_elapsed_time' in metrics else ""
        kw_str += f'''{f"Epoch dur: {utils.format_time(metrics['epoch_time'], 2)}":25s} ''' if 'epoch_time' in metrics else ""
        kw_str += f'''{f"avg/step: {utils.format_time(metrics['step_avg_time'], 2)}":25s} ''' if 'step_avg_time' in metrics else ""
        kw_str += f'''{f"loss: {metrics['loss']:7.4f}":12s} ''' if 'loss' in metrics else ""
        kw_str += f'''{f"acc: {metrics['accuracy']:4.2%}":12s} ''' if 'accuracy' in metrics else ""
        kw_str += f'''{f"auc: {metrics['auc']:7.4f}":12s} ''' if 'auc' in metrics else ""
        kw_str += f'''{f"f1: {metrics['f1']:7.4f}":12s} ''' if 'f1' in metrics else ""
        # kw_str += f'''{f"acc consol: {metrics['accuracy_consolidated']:4.2%}":12s} ''' if 'accuracy_consolidated' in metrics else ""
        # kw_str += f'''{f"auc consol: {metrics['auc_consolidated']:7.4f}":12s} ''' if 'auc_consolidated' in metrics else ""
        # kw_str += f'''{f"f1 consol: {metrics['f1_consolidated']:7.4f}":12s} ''' if 'f1_consolidated' in metrics else ""
        if metrics.get("VRAM", 0.0) > 0.0:
            kw_str += f'''{f"Mem (VRAM/RAM): {metrics['VRAM']:5.4f}GB/{metrics['RAM']:5.4f}":15s}GB '''
            # kw_str += f'''{f"Mem (USAGE): {metrics['Usage']:5.4f}GB/{metrics['Total']:5.4f}":15s}GB '''
        else:
            kw_str += f'''{f"Mem (RAM): {metrics['RAM']:4.2f}":15s} ''' if 'RAM' in metrics else ""

        log.info(f"{message:25s} | {kw_str}")

        if self.setup.get("wandb_enabled", False):
            logged = ["batches", "epoch_time", "total_elapsed_time", "step_avg_time", "loss", "accuracy", "auc", "f1"]
            prefix = "train_" if is_training else "eval_"
            wandb_metrics = {f"{prefix}{k}": v for k, v in metrics.items() if k in logged}
            wandb_metrics["epoch"] = self.epoch
            wandb_metrics["step"] = self.steps
            # print("wandb", wandb_metrics)
            utils.wandb_log(wandb_metrics)

    def _init_distributed(self, model):
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[self.setup["device"]] if self.setup["device"].type == "cuda" else None,
            output_device=self.setup["device"] if self.setup["device"].type == "cuda" else None,
            broadcast_buffers=self.cfg_impl.broadcast_buffers,
            bucket_cap_mb=self.cfg_impl.bucket_cap_mb,
            gradient_as_bucket_view=self.cfg_impl.gradient_as_bucket_view,
            static_graph=self.cfg_impl.static_graph,
        )
        return model
