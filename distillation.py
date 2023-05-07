#!/usr/bin/env python


import argparse
import json
import glob
import gc
import os
import sys
import warnings
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union, Callable,  Iterable
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from transformers.utils import logging
# from finetune import SummarizationModule, BaseTransformer
# from finetune import main as ft_main
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoModelForSeq2SeqLM,  T5ForConditionalGeneration, AutoTokenizer, PreTrainedModel
from transformers.models.bart.modeling_bart import shift_tokens_right
from utils import freeze_params, pickle_save

#  -=--------------------------------------------------Fine tune starts here ----------------------------------------------------
import time
from collections import defaultdict
import numpy as np
from torch.utils.data import DataLoader
import logging
from transformers.utils.versions import require_version
from transformers import (
    AdamW,
    AutoConfig,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoModelWithLMHead,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    MBartTokenizer,
    T5ForConditionalGeneration
)
from transformers.optimization import (
    Adafactor,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from utils import (
    ROUGE_KEYS,
    LegacySeq2SeqDataset,
    Seq2SeqDataset,
    assert_all_frozen,
    calculate_rouge,
    flatten_list,

)


args = {
    "accelerator": None,
    "accumulate_grad_batches": 32,
    "adafactor": False,
    "adam_epsilon": 1.0e-08,
    "alpha_ce": 0.8,
    "alpha_encoder_loss": 0.0,
    "alpha_hid": 0.0,
    "alpha_mlm": 0.2,
    "amp_backend": "native",
    "amp_level": "O2",
    "attention_dropout": None,
    "auto_lr_find": False,
    "auto_scale_batch_size": False,
    "auto_select_gpus": False,
    "automatic_optimization": True,
    "benchmark": False,
    "cache_dir": "",
    "check_val_every_n_epoch": 1,
    "checkpoint_callback": True,
    "config_name": "",
    "data_dir": "Data",
    "decoder_layerdrop": None,
    "default_root_dir": None,
    "deterministic": False,
    "distributed_backend": None,
    "do_predict": True,
    "do_train": True,
    "dropout": None,
    "early_stopping_patience": -1,
    "encoder_layerdrop": None,
    "eval_batch_size": 1,
    "eval_beams": None,
    "eval_max_gen_length": None,
    "fast_dev_run": False,
    "flush_logs_every_n_steps": 100,
    "fp16": False,
    "fp16_opt_level": "O2",
    "freeze_embeds": True,
    "freeze_encoder": True,
    "gpus": 0,
    "gradient_clip_val": 0,
    "label_smoothing": 0.0,
    "learning_rate": 5.0e-05,
    "length_penalty": -1,
    "limit_test_batches": 1.0,
    "limit_train_batches": 1.0,
    "limit_val_batches": 1.0,
    "log_every_n_steps": 50,
    "log_gpu_memory": None,
    "logger": False,
    "logger_name": "default",
    "lr_scheduler": "linear",
    "max_epochs": 1,
    "max_source_length": 1024,
    "max_steps": None,
    "max_target_length": 56,
    "max_tokens_per_batch": None,
    "min_epochs": 1,
    "min_steps": None,
    "model_name_or_path": "google/pegasus-cnn_dailymail",
    "n_test": 1,
    "n_train": 10,
    "n_val": 1,
    "no_teacher": True,
    "normalize_hidden": False,
    "num_nodes": 1,
    "num_processes": 1,
    "num_sanity_val_steps": 2,
    "num_workers": 0,
    "output_dir": "Result_weights",
    "overfit_batches": 0.0,
    "precision": 32,
    "prepare_data_per_node": True,
    "process_position": 0,
    "profiler": None,
    "progress_bar_refresh_rate": 1,
    "reload_dataloaders_every_epoch": False,
    "replace_sampler_ddp": True,
    "resume_from_checkpoint": None,
    "save_top_k": 3,
    "seed": 42,
    "setup_cls": "SummarizationModule",
    "sortish_sampler": False,
    "src_lang": "",
    "student_decoder_layers": 4,
    "student_encoder_layers": 12,
    "supervise_forward": False,
    "sync_batchnorm": False,
    "task": "summarization",
    "teacher": None,
    "terminate_on_nan": False,
    "test_max_target_length": 142,
    "tgt_lang": "",
    "tokenizer_name": None,
    # "tpu_cores": '',
    # "n-tpu_cores": '',
    "track_grad_norm": -1,
    "train_batch_size": 1,
    "truncated_bptt_steps": None,
    "val_check_interval": 0.1,
    "val_max_target_length": 142,
    "val_metric": None,
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "weights_save_path": "Results/weights",
    "weights_summary": "top"
}


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


logger = logging.getLogger(__name__)

require_version("pytorch_lightning>=1.0.4")

MODEL_MODES = {
    "base": AutoModel,
    "sequence-classification": AutoModelForSequenceClassification,
    "question-answering": AutoModelForQuestionAnswering,
    "pretraining": AutoModelForPreTraining,
    "token-classification": AutoModelForTokenClassification,
    "language-modeling": AutoModelWithLMHead,
    "summarization": AutoModelForSeq2SeqLM,
    "translation": AutoModelForSeq2SeqLM,
}


# update this and the import above to support new schedulers from transformers.optimization
arg_to_scheduler = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    # '': get_constant_schedule,             # not supported for now
    # '': get_constant_schedule_with_warmup, # not supported for now
}
arg_to_scheduler_choices = sorted(arg_to_scheduler.keys())
arg_to_scheduler_metavar = "{" + ", ".join(arg_to_scheduler_choices) + "}"


class BaseTransformer(pl.LightningModule):
    def __init__(
        self,
        hparams: argparse.Namespace,
        num_labels=None,
        mode="base",
        config=None,
        tokenizer=None,
        model=None,
        **config_kwargs,
    ):
        """Initialize a model, tokenizer and config."""
        super().__init__()

        self.save_hyperparameters(hparams)
        self.step_count = 0
        self.output_dir = Path(self.hparams["output_dir"])
        cache_dir = self.hparams["cache_dir"] if self.hparams["cache_dir"] else None
        if config is None:
            self.config = AutoConfig.from_pretrained(
                self.hparams["config_name"] if self.hparams["config_name"] else self.hparams["model_name_or_path"],
                **({"num_labels": num_labels} if num_labels is not None else {}),
                cache_dir=cache_dir,
                **config_kwargs,
            )
        else:
            self.config: PretrainedConfig = config

        extra_model_params = (
            "encoder_layerdrop", "decoder_layerdrop", "dropout", "attention_dropout")
        for p in extra_model_params:
            if getattr(self.hparams, p, None):
                assert hasattr(
                    self.config, p), f"model config doesn't have a `{p}` attribute"
                setattr(self.config, p, getattr(self.hparams, p))

        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.hparams["tokenizer_name"] if self.hparams["tokenizer_name"] else self.hparams["model_name_or_path"],
                cache_dir=cache_dir,
            )
        else:
            self.tokenizer: PreTrainedTokenizer = tokenizer
        self.model_type = MODEL_MODES[mode]
        if model is None:
            self.model = self.model_type.from_pretrained(
                self.hparams["model_name_or_path"],
                from_tf=bool(".ckpt" in self.hparams["model_name_or_path"]),
                config=self.config,
                cache_dir=cache_dir,
            )
        else:
            self.model = model

    def load_hf_checkpoint(self, *args, **kwargs):
        self.model = self.model_type.from_pretrained(*args, **kwargs)

    def get_lr_scheduler(self):
        get_schedule_func = arg_to_scheduler[self.hparams["lr_scheduler"]]
        scheduler = get_schedule_func(
            self.opt, num_warmup_steps=self.hparams["warmup_steps"], num_training_steps=self.total_steps(
            )
        )
        scheduler = {"scheduler": scheduler,
                     "interval": "step", "frequency": 1}
        return scheduler

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        if self.hparams["adafactor"]:
            optimizer = Adafactor(
                optimizer_grouped_parameters, lr=self.hparams["learning_rate"], scale_parameter=False, relative_step=False
            )

        else:
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=self.hparams[
                    "learning_rate"], eps=self.hparams["adam_epsilon"]
            )
        self.opt = optimizer

        scheduler = self.get_lr_scheduler()

        return [optimizer], [scheduler]

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        return self.validation_end(outputs)

    def total_steps(self) -> int:
        """The number of total training steps that will be run. Used for lr scheduler purposes."""
        num_devices = max(
            1, self.hparams["gpus"])  # TODO: consider num_tpu_cores
        effective_batch_size = self.hparams["train_batch_size"] * \
            self.hparams["accumulate_grad_batches"] * num_devices
        return (self.dataset_size / effective_batch_size) * self.hparams["max_epochs"]

    def setup(self, mode):
        if mode == "test":
            self.dataset_size = len(self.test_dataloader().dataset)
        else:
            self.train_loader = self.get_dataloader(
                "train", self.hparams["train_batch_size"], shuffle=True)
            self.dataset_size = len(self.train_dataloader().dataset)

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False):
        raise NotImplementedError("You must implement this for your task")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.get_dataloader("dev", self.hparams["eval_batch_size"], shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader("test", self.hparams["eval_batch_size"], shuffle=False)

    def _feature_file(self, mode):
        return os.path.join(
            self.hparams["data_dir"],
            "cached_{}_{}_{}".format(
                mode,
                list(
                    filter(None, self.hparams["model_name_or_path"].split("/"))).pop(),
                str(self.hparams["max_seq_length"]),
            ),
        )

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = self.output_dir.joinpath("best_tfmr")
        self.model.config.save_step = self.step_count
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)


sys.path.insert(2, str(Path(__file__).resolve().parents[1]))


class SummarizationModule(BaseTransformer):
    mode = "summarization"
    loss_names = ["loss"]
    metric_names = ROUGE_KEYS
    default_val_metric = "rouge2"

    def __init__(self, hparams, **kwargs):
        if hparams["sortish_sampler"] and hparams["gpus"] > 1:
            hparams["replace_sampler_ddp"] = False
        elif hparams["max_tokens_per_batch"] is not None:
            if hparams["gpus"] > 1:
                raise NotImplementedError(
                    "Dynamic Batch size does not work for multi-gpu training")
            if hparams["sortish_sampler"]:
                raise ValueError(
                    "--sortish_sampler and --max_tokens_per_batch may not be used simultaneously")

        super().__init__(hparams, num_labels=None, mode=self.mode, **kwargs)
        use_task_specific_params(self.model, "summarization")
        # save_git_info(self.hparams["output_dir"])
        self.metrics_save_path = Path(self.output_dir) / "metrics.json"
        self.hparams_save_path = Path(self.output_dir) / "hparams.pkl"
        pickle_save(self.hparams, self.hparams_save_path)
        self.step_count = 0
        self.metrics = defaultdict(list)
        self.model_type = self.config.model_type
        self.vocab_size = self.config.tgt_vocab_size if self.model_type == "fsmt" else self.config.vocab_size

        self.dataset_kwargs: dict = dict(
            data_dir=self.hparams["data_dir"],
            max_source_length=self.hparams["max_source_length"],
            prefix=self.model.config.prefix or "",
        )
        n_observations_per_split = {
            "train": self.hparams["n_train"],
            "val": self.hparams["n_val"],
            "test": self.hparams["n_test"],
        }
        self.n_obs = {k: v if v >= 0 else None for k,
                      v in n_observations_per_split.items()}

        self.target_lens = {
            "train": self.hparams["max_target_length"],
            "val": self.hparams["val_max_target_length"],
            "test": self.hparams["test_max_target_length"],
        }
        assert self.target_lens["train"] <= self.target_lens[
            "val"], f"target_lens: {self.target_lens}"
        assert self.target_lens["train"] <= self.target_lens[
            "test"], f"target_lens: {self.target_lens}"
        if self.hparams["freeze_embeds"]:
            freeze_embeds(self.model)
        if self.hparams["freeze_encoder"]:
            freeze_params(self.model.get_encoder())
            assert_all_frozen(self.model.get_encoder())

        #self.hparams.git_sha = get_git_info()["repo_sha"]
        self.num_workers = hparams["num_workers"]
        self.decoder_start_token_id = None  # default to config
        if self.model.config.decoder_start_token_id is None and isinstance(self.tokenizer, MBartTokenizer):
            self.decoder_start_token_id = self.tokenizer.lang_code_to_id[hparams["tgt_lang"]]
            self.model.config.decoder_start_token_id = self.decoder_start_token_id
        self.dataset_class = (
            Seq2SeqDataset if hasattr(
                self.tokenizer, "prepare_seq2seq_batch") else LegacySeq2SeqDataset
        )
        self.eval_beams = self.model.config.num_beams if self.hparams[
            "eval_beams"] is None else self.hparams["eval_beams"]
        if self.hparams["eval_max_gen_length"] is not None:
            self.eval_max_length = self.hparams["eval_max_gen_length"]
        else:
            self.eval_max_length = self.model.config.max_length
        self.val_metric = self.default_val_metric if self.hparams[
            "val_metric"] is None else self.hparams["val_metric"]

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    def ids_to_clean_text(self, generated_ids: List[int]):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return lmap(str.strip, gen_text)

    def _step(self, batch: dict) -> Tuple:
        pad_token_id = self.tokenizer.pad_token_id
        src_ids, src_mask = batch["input_ids"], batch["attention_mask"]
        tgt_ids = batch["labels"]
        if isinstance(self.model, T5ForConditionalGeneration):
            decoder_input_ids = self.model._shift_right(tgt_ids)
        else:
            decoder_input_ids = shift_tokens_right(tgt_ids, pad_token_id)

        outputs = self(src_ids, attention_mask=src_mask,
                       decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        if self.hparams["label_smoothing"] == 0:
            # Same behavior as modeling_bart.py, besides ignoring pad_token_id
            ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

            assert lm_logits.shape[-1] == self.vocab_size
            loss = ce_loss_fct(
                lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))
        else:
            lprobs = torch.nn.functional.log_softmax(lm_logits, dim=-1)
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, tgt_ids, self.hparams["label_smoothing"], ignore_index=pad_token_id
            )
        return (loss,)

    @property
    def pad(self) -> int:
        return self.tokenizer.pad_token_id

    def training_step(self, batch, batch_idx) -> Dict:
        loss_tensors = self._step(batch)

        logs = {name: loss for name, loss in zip(
            self.loss_names, loss_tensors)}
        # tokens per batch
        logs["tpb"] = batch["input_ids"].ne(
            self.pad).sum() + batch["labels"].ne(self.pad).sum()
        logs["bs"] = batch["input_ids"].shape[0]
        logs["src_pad_tok"] = batch["input_ids"].eq(self.pad).sum()
        logs["src_pad_frac"] = batch["input_ids"].eq(self.pad).float().mean()
        # TODO(SS): make a wandb summary metric for this
        return {"loss": loss_tensors[0], "log": logs}

    def validation_step(self, batch, batch_idx) -> Dict:
        return self._generative_step(batch)

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.step_count += 1
        losses = {k: torch.stack([x[k] for x in outputs]).mean()
                  for k in self.loss_names}
        loss = losses["loss"]
        generative_metrics = {
            k: np.array([x[k] for x in outputs]).mean() for k in self.metric_names + ["gen_time", "gen_len"]
        }
        metric_val = (
            generative_metrics[self.val_metric] if self.val_metric in generative_metrics else losses[self.val_metric]
        )
        metric_tensor: torch.FloatTensor = torch.tensor(
            metric_val).type_as(loss)
        generative_metrics.update({k: v.item() for k, v in losses.items()})
        losses.update(generative_metrics)
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["step_count"] = self.step_count
        # callback writes this to self.metrics_save_path
        self.metrics[prefix].append(all_metrics)
        preds = flatten_list([x["preds"] for x in outputs])
        return {
            "log": all_metrics,
            "preds": preds,
            f"{prefix}_loss": loss,
            f"{prefix}_{self.val_metric}": metric_tensor,
        }

    def calc_generative_metrics(self, preds, target) -> Dict:
        return calculate_rouge(preds, target)

    def _generative_step(self, batch: dict) -> dict:
        t0 = time.time()

        # parser.add_argument('--eval_max_gen_length', type=int, default=None, help='never generate more than n tokens')
        generated_ids = self.model.generate(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=True,
            decoder_start_token_id=self.decoder_start_token_id,
            num_beams=self.eval_beams,
            max_length=self.eval_max_length,
        )
        gen_time = (time.time() - t0) / batch["input_ids"].shape[0]
        preds: List[str] = self.ids_to_clean_text(generated_ids)
        target: List[str] = self.ids_to_clean_text(batch["labels"])
        loss_tensors = self._step(batch)
        base_metrics = {name: loss for name,
                        loss in zip(self.loss_names, loss_tensors)}
        rouge: Dict = self.calc_generative_metrics(preds, target)
        summ_len = np.mean(lmap(len, generated_ids))
        base_metrics.update(gen_time=gen_time, gen_len=summ_len,
                            preds=preds, target=target, **rouge)
        return base_metrics

    def test_step(self, batch, batch_idx):
        return self._generative_step(batch)

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")

    def get_dataset(self, type_path) -> Seq2SeqDataset:
        n_obs = self.n_obs[type_path]
        max_target_length = self.target_lens[type_path]
        dataset = self.dataset_class(
            self.tokenizer,
            type_path=type_path,
            n_obs=n_obs,
            max_target_length=max_target_length,
            **self.dataset_kwargs,
        )
        return dataset

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        dataset = self.get_dataset(type_path)

        if self.hparams["sortish_sampler"] and type_path != "test":
            sampler = dataset.make_sortish_sampler(
                batch_size, distributed=self.hparams["gpus"] > 1)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=False,
                num_workers=self.num_workers,
                sampler=sampler,
            )

        elif self.hparams["max_tokens_per_batch"] is not None and type_path != "test":
            batch_sampler = dataset.make_dynamic_sampler(
                self.hparams["max_tokens_per_batch"], distributed=self.hparams["gpus"] > 1
            )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=dataset.collate_fn,
                # shuffle=False,
                num_workers=self.num_workers,
                # batch_size=None,
            )
        else:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=dataset.collate_fn,
                shuffle=shuffle,
                num_workers=self.num_workers,
                sampler=None,
            )

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader(
            "train", batch_size=self.hparams["train_batch_size"], shuffle=True)
        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("val", batch_size=self.hparams["eval_batch_size"])

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.hparams["eval_batch_size"])


#  -=--------------------------------------------------Fine tune ends here ----------------------------------------------------


def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"using task specific params for {task}: {pars}")
        model.config.update(pars)


# util ends here


# logger = logging.get_logger(__name__)


def copy_layers(src_layers: nn.ModuleList, dest_layers: nn.ModuleList, layers_to_copy: List[int]) -> None:
    layers_to_copy = nn.ModuleList(
        [l for i, l in enumerate(src_layers) if i in layers_to_copy])
    assert len(dest_layers) == len(
        layers_to_copy), f"{len(dest_layers)} != {len(layers_to_copy)}"
    dest_layers.load_state_dict(layers_to_copy.state_dict())


LAYERS_TO_COPY = {
    # maps  num layers in teacher -> num_layers in student -> which teacher layers to copy.
    # 12: bart, 16: pegasus, 6: marian/Helsinki-NLP
    12: {
        1: [0],  # This says that if the teacher has 12 layers and the student has 1, copy layer 0 of the teacher
        2: [0, 6],
        3: [0, 6, 11],
        4: [0, 4, 8, 11],
        6: [0, 2, 4, 7, 9, 11],
        9: [0, 1, 2, 4, 5, 7, 9, 10, 11],
        12: list(range(12)),
    },
    16: {  # maps  num layers in student -> which teacher layers to copy
        1: [0],
        2: [0, 15],
        3: [0, 8, 15],
        4: [0, 5, 10, 15],
        6: [0, 3, 6, 9, 12, 15],
        8: [0, 2, 4, 6, 8, 10, 12, 15],
        9: [0, 1, 3, 5, 7, 9, 11, 13, 15],
        12: [0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 13, 15],
        16: list(range(16)),
    },
    6: {1: [0], 2: [0, 5], 3: [0, 2, 5], 4: [0, 1, 3, 5], 6: list(range(6))},
}
LAYERS_TO_SUPERVISE = {
    # maps  num layers in student -> which teacher layers to copy.
    6: {1: [5], 2: [3, 5], 3: [1, 4, 5], 4: [1, 2, 4, 5]},
    12: {1: [11], 2: [5, 11], 3: [3, 7, 11], 6: [1, 3, 5, 8, 10, 11]},
    16: {1: [15], 4: [4, 9, 12, 15], 8: [1, 3, 5, 7, 9, 11, 13, 15]},
}


def pick_layers_to_copy(n_student, n_teacher):
    try:
        val = LAYERS_TO_COPY[n_teacher][n_student]
        return val
    except KeyError:
        if n_student != n_teacher:
            warnings.warn(
                f"no hardcoded layers to copy for teacher {n_teacher} -> student {n_student}, defaulting to first {n_student}"
            )
        return list(range(n_student))


def get_layers_to_supervise(n_student, n_teacher) -> List[int]:
    """Used or the --supervise_forward kwarg"""
    if n_student > n_teacher:
        raise ValueError(
            f"Cannot perform intermediate supervision for student {n_student} > teacher {n_teacher}")
    elif n_teacher == n_student:
        return list(range(n_teacher))
    elif n_student == 1:
        return [n_teacher - 1]
    else:
        return LAYERS_TO_SUPERVISE[n_teacher][n_student]


def create_student_by_copying_alternating_layers(
    teacher: Union[str, PreTrainedModel],
    save_path: Union[str, Path] = "student",
    e: Union[int, None] = None,
    d: Union[int, None] = None,
    copy_first_teacher_layers=False,
    **extra_config_kwargs
) -> Tuple[PreTrainedModel, List[int], List[int]]:

    _msg = "encoder_layers and decoder_layers cannot be both None-- you would just have an identical teacher."
    assert (e is not None) or (d is not None), _msg
    if isinstance(teacher, str):
        AutoTokenizer.from_pretrained(teacher).save_pretrained(
            save_path)  # purely for convenience
        teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher).eval()
    else:

        assert isinstance(
            teacher, PreTrainedModel), f"teacher must be a model or string got type {type(teacher)}"
    init_kwargs = teacher.config.to_diff_dict()

    try:
        teacher_e, teacher_d = teacher.config.encoder_layers, teacher.config.decoder_layers
        if e is None:
            e = teacher_e
        if d is None:
            d = teacher_d
        init_kwargs.update({"encoder_layers": e, "decoder_layers": d})
    except AttributeError:  # T5
        teacher_e, teacher_d = teacher.config.num_layers, teacher.config.num_decoder_layers
        if e is None:
            e = teacher_e
        if d is None:
            d = teacher_d
        init_kwargs.update({"num_layers": e, "num_decoder_layers": d})

    # Kwargs to instantiate student: teacher kwargs with updated layer numbers + **extra_config_kwargs
    init_kwargs.update(extra_config_kwargs)

    # Copy weights
    student_cfg = teacher.config_class(**init_kwargs)
    student = AutoModelForSeq2SeqLM.from_config(student_cfg)
    # Start by copying the full teacher state dict this will copy the first N teacher layers to the student.
    info = student.load_state_dict(teacher.state_dict(), strict=False)
    # every student key should have a teacher keys.
    assert info.missing_keys == [], info.missing_keys

    if copy_first_teacher_layers:  # Our copying is done. We just log and save
        e_layers_to_copy, d_layers_to_copy = list(range(e)), list(range(d))
        logger.info(
            f"Copied encoder layers {e_layers_to_copy} and decoder layers {d_layers_to_copy}. Saving them to {save_path}"
        )
        student.save_pretrained(save_path)
        return student, e_layers_to_copy, d_layers_to_copy

    # Decide which layers of the teacher to copy. Not exactly alternating -- we try to keep first and last layer.
    e_layers_to_copy: List[int] = pick_layers_to_copy(e, teacher_e)
    d_layers_to_copy: List[int] = pick_layers_to_copy(d, teacher_d)

    try:
        copy_layers(teacher.model.encoder.layers,
                    student.model.encoder.layers, e_layers_to_copy)
        copy_layers(teacher.model.decoder.layers,
                    student.model.decoder.layers, d_layers_to_copy)
    except AttributeError:  # For t5, student.model.encoder.layers is called student.encoder.block
        copy_layers(teacher.encoder.block,
                    student.encoder.block, e_layers_to_copy)
        copy_layers(teacher.decoder.block,
                    student.decoder.block, d_layers_to_copy)
    logger.info(
        f"Copied encoder layers {e_layers_to_copy} and decoder layers {d_layers_to_copy}. Saving them to {save_path}"
    )
    student.config.init_metadata = dict(
        teacher_type=teacher.config.model_type,
        copied_encoder_layers=e_layers_to_copy,
        copied_decoder_layers=d_layers_to_copy,
    )
    student.save_pretrained(save_path)
    # Save information about copying for easier reproducibility

    return student, e_layers_to_copy, d_layers_to_copy


def get_early_stopping_callback(metric, patience):
    return EarlyStopping(
        monitor=f"val_{metric}",  # does this need avg?
        mode="min" if "loss" in metric else "max",
        patience=patience,
        verbose=True,
    )


def get_checkpoint_callback(output_dir, metric, save_top_k=1, lower_is_better=False):
    """Saves the best model by validation ROUGE2 score."""
    if metric == "rouge2":
        exp = "{val_avg_rouge2:.4f}-{step_count}"
    elif metric == "bleu":
        exp = "{val_avg_bleu:.4f}-{step_count}"
    elif metric == "loss":
        exp = "{val_avg_loss:.4f}-{step_count}"
    else:
        raise NotImplementedError(
            f"seq2seq callbacks only support rouge2, bleu and loss, got {metric}, You can make your own by adding to this function."
        )

    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(output_dir, exp),
        monitor=f"val_{metric}",
        mode="min" if "loss" in metric else "max",
        save_top_k=save_top_k,
        # maybe save a checkpoint every time val is run, not just end of epoch.
        period=0,
    )
    return checkpoint_callback


def create_module(args):
    module_cls = SummarizationModule
    args["setup_cls"]: str = module_cls.__name__
    print(f'using module {args["setup_cls"]}')
    model = module_cls(args)
    return model


def generic_train(
    model: BaseTransformer,
    args,
    early_stopping_callback=None,
    # logger=True,  # can pass WandbLogger() here
    logger=False,
    extra_callbacks=[],
    checkpoint_callback=None,
    logging_callback=None,
    **extra_train_kwargs,
):
    pl.seed_everything(args["seed"])

    # init model
    odir = Path(model.hparams["output_dir"])
    odir.mkdir(exist_ok=True)

    # add custom checkpoints
    if checkpoint_callback is None:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            filepath=args["output_dir"], prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
        )
    if early_stopping_callback:
        extra_callbacks.append(early_stopping_callback)

    train_params = {}

    # TODO: remove with PyTorch 1.6 since pl uses native amp
    if args["fp16"]:
        train_params["precision"] = 16
        train_params["amp_level"] = args["fp16_opt_level"]

    if args["gpus"] > 1:
        train_params["distributed_backend"] = "ddp"

    train_params["accumulate_grad_batches"] = args["accumulate_grad_batches"]
    train_params["accelerator"] = extra_train_kwargs.get("accelerator", None)
    train_params["profiler"] = extra_train_kwargs.get("profiler", None)

    args1 = argparse.Namespace(**args)

    trainer = pl.Trainer.from_argparse_args(
        # args,
        args1,
        # weights_summary=None,

        # callbacks=[logging_callback] + extra_callbacks,
        # logger=logger,
        # checkpoint_callback=checkpoint_callback,
        ** train_params,
    )

    if args["do_train"]:
        trainer.fit(model)

    return trainer


def evaluate_checkpoint(ckpt_path: Path, dest_dir=None):
    # TODO(SS): DELETE? Better to convert_pl_ckpt_to_hf and run_eval.py
    exp_dir = ckpt_path.parent
    if dest_dir is None:
        dest_dir = exp_dir
    clash = list(dest_dir.glob("test_generations*"))
    if clash:
        print(f"SKIPPING to avoid overwriting {clash}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "hparams" in ckpt:
        args = argparse.Namespace(**ckpt["hparams"])
    else:
        args = argparse.Namespace(**pickle_load(exp_dir / "hparams.pkl"))
    args["resume_from_checkpoint"] = str(ckpt_path)
    args["do_train"] = False
    args["output_dir"] = str(dest_dir)
    args["n_gpu"] = 1
    args["eval_batch_size"] = 16
    Path(args["output_dir"]).mkdir(exist_ok=True)
    model = create_module(args)
    trainer: pl.Trainer = generic_train(
        model, args, early_stopping_callback=False)
    trainer.test(model)


def main(args) -> SummarizationModule:
    Path(args["output_dir"]).mkdir(exist_ok=True)
    if len(os.listdir(args["output_dir"])) > 3 and args["do_train"]:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args["output_dir"]))

    model = create_module(args)
    print("new_main")
    Path(args["output_dir"]).mkdir(exist_ok=True)
    if len(os.listdir(args["output_dir"])) > 3 and args["do_train"]:
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(args["output_dir"]))
    if model is None:
        if "summarization" in args["task"]:
            model: SummarizationModule = SummarizationModule(args)

    if args["early_stopping_patience"] >= 0:
        es_callback = get_early_stopping_callback(
            model.val_metric, args["early_stopping_patience"])
    else:
        es_callback = False

    lower_is_better = args["val_metric"] == "loss"
    trainer: pl.Trainer = generic_train(
        model,
        args,
        checkpoint_callback=get_checkpoint_callback(
            args["output_dir"], model.val_metric, args["save_top_k"], lower_is_better
        ),
        early_stopping_callback=es_callback,
    )
    pickle_save(model.hparams, model.output_dir / "hparams.pkl")
    if not args["do_predict"]:
        return model

    model.hparams["test_checkpoint"] = ""
    checkpoints = list(
        sorted(glob.glob(os.path.join(args["output_dir"], "*.ckpt"), recursive=True)))
    if checkpoints:
        model.hparams["test_checkpoint"] = checkpoints[-1]
        trainer.resume_from_checkpoint = checkpoints[-1]
    trainer.logger.log_hyperparams(model.hparams)

    # test() without a model tests using the best checkpoint automatically
    trainer.test()
    return model


if __name__ == "__main__":
    main(args)
