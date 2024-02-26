import math
import os
import random
from dataclasses import dataclass
from functools import partial
from typing import Any, Optional, Union

import numpy as np
import torch
from nltk import sent_tokenize
from rouge_score import rouge_scorer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


def read_slurm_env():
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    devices = int(os.environ["SLURM_GPUS_ON_NODE"])
    num_nodes = int(os.environ["SLURM_NNODES"])
    return rank, local_rank, world_size, devices, num_nodes


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def decode_batch_labels(tokenizer, batch_labels):
    decoded = []
    for target in batch_labels:
        target = target[target != tokenizer.pad_token_id]
        sentence = tokenizer.decode(target, skip_special_tokens=True)
        sentence = "\n".join(sent_tokenize(sentence))
        decoded.append(sentence)
    return decoded


def compute_loss(model, eval_dataloader):
    loss = 0.0
    for n_iter, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            batch = {k: v.cuda() for k, v in batch.items()}
            out = model(**batch)
            loss += out.loss.detach().item()
        return loss / (n_iter + 1)


def eval_rouge(list_predictions, list_targets, n_print=0, reduce=True):
    rouge1, rouge2, rougeLsum, meanrouge = [], [], [], []
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeLsum"], use_stemmer=False
    )
    for pred, target in zip(list_predictions, list_targets):
        results = scorer.score(prediction=pred, target=target)
        rougeLsum.append(results["rougeLsum"].fmeasure)
        rouge1.append(results["rouge1"].fmeasure)
        rouge2.append(results["rouge2"].fmeasure)
        meanrouge.append(
            (
                results["rouge1"].fmeasure
                + results["rouge2"].fmeasure
                + results["rougeLsum"].fmeasure
            )
            / 3.0
        )
    if reduce:
        dict_results = {
            "rouge1": np.mean(rouge1),
            "rouge2": np.mean(rouge2),
            "rougeLsum": np.mean(rougeLsum),
            "mean-rouge": np.mean(meanrouge),
        }
    else:
        dict_results = {
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeLsum": rougeLsum,
            "mean-rouge": meanrouge,
        }

    if n_print > 0:
        indices = [i for i in range(n_print)]
        for i in indices:
            print("Source:", list_targets[i])
            print()
            print("Prediction:", list_predictions[i])
            print()
    return dict_results


def get_scheduler(scheduler_name="cosine", **kwargs):
    if scheduler_name == "cosine":
        return get_cosine_schedule_with_warmup(**kwargs)
    if scheduler_name == "square-root":
        return get_inverse_power_schedule_with_warmup(**kwargs)
    if scheduler_name == "t5":
        return get_longt5_scheduler(**kwargs)

    if scheduler_name == "constant":
        return get_constant_schedule(**kwargs)
    if scheduler_name == "constant-warmup":
        return get_constant_schedule_with_warmup(**kwargs)


def get_inverse_power_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps=None,
    num_plateau_steps=0,
    lr_plateau=1e-3,
    power=0.5,
    last_epoch=-1,
    **kwargs,
):
    lr_init = optimizer.defaults["lr"]
    lr_end = (
        (
            lr_init
            * (num_warmup_steps) ** power
            * (num_training_steps - num_plateau_steps) ** (-power)
        )
        if num_training_steps is not None
        else None
    )

    def lr_lambda(current_step: int):
        if current_step < num_plateau_steps:
            return lr_plateau / lr_init
        if current_step < num_plateau_steps + num_warmup_steps:
            return float(current_step - num_plateau_steps) / float(
                max(1, num_warmup_steps)
            )
        elif current_step >= num_training_steps and lr_end is not None:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr = max(num_warmup_steps, 1) ** power * (
                max(1, current_step - num_plateau_steps) ** (-power)
            )
        return lr

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_optimizer(optimizer_name):
    if optimizer_name == "adamw":
        try:
            # default to fused AdamW if apex is installed
            # based on this benchmark https://github.com/huggingface/transformers/issues/22101
            from apex.optimizers import FusedAdam

            optimizer_cls = FusedAdam
        except:
            from transformers import AdamW

            optimizer_cls = AdamW
            optimizer_cls = partial(optimizer_cls, betas=(0.9, 0.999), weight_decay=0.0)
    elif optimizer_name == "adafactor":
        from transformers import Adafactor

        optimizer_cls = partial(
            Adafactor, clip_threshold=1.0, scale_parameter=False, relative_step=False
        )
    return optimizer_cls


def get_longt5_scheduler(optimizer, num_warmup_steps, **kwargs):
    def lr_lambda(current_step: int):
        factor = math.sqrt(num_warmup_steps + 1)
        return factor / math.sqrt(max(current_step + 1, num_warmup_steps + 1))

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


def get_constant_schedule(optimizer, last_epoch: int = -1, **kwargs):
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    last_epoch: int = -1,
    **kwargs,
):
    lr_init = optimizer.defaults["lr"]

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            return 1

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
    lr_end=1e-6,
):
    lr_init = optimizer.defaults["lr"]

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step > num_training_steps:
            return lr_end / lr_init
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return (
            max(
                lr_end,
                0.5
                * (lr_init - lr_end)
                * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
                + lr_end,
            )
            / lr_init
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_special_lr_params(model):
    special_lr = []
    normal_lr = []
    for p in model.parameters():
        if p.requires_grad:
            if hasattr(p, "_optim"):
                if p._optim.get("special_lr", None):
                    special_lr.append(p)
                else:
                    normal_lr.append(p)
            else:
                normal_lr.append(p)

    return special_lr, normal_lr


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    truncate: bool = False

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (
                    max_label_length - len(feature["labels"])
                )
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder
                        if padding_side == "right"
                        else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate(
                        [feature["labels"], remainder]
                    ).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate(
                        [remainder, feature["labels"]]
                    ).astype(np.int64)
                if self.truncate:
                    source_feature = feature["input_ids"]
                    if len(source_feature) > self.max_length:
                        source_feature = source_feature[: self.max_length - 1]
                        if isinstance(source_feature, list):
                            source_feature = source_feature + [
                                self.tokenizer.eos_token_id
                            ]
                        else:
                            source_feature = np.concatenate(
                                (source_feature, [self.tokenizer.eos_token_id]),
                            ).astype(np.int64)
                        feature["input_ids"] = source_feature
                        feature["attention_mask"] = feature["attention_mask"][
                            : len(source_feature)
                        ]

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(
                labels=features["labels"]
            )
            features["decoder_input_ids"] = decoder_input_ids

        return features
