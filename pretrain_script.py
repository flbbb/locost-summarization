import argparse
import os
import re
from pathlib import Path

import pytorch_lightning as pl
import torch
from datasets.load import load_from_disk
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models import LOCOSTForConditionalGeneration
from models_config import LOCOSTConfig
from models_lightning import LitLLMForConditionalGeneration
from utils import DataCollator, read_slurm_env

load_dotenv()
DATASET_PATH = Path(os.environ["DATASET_PATH"])
TOKENIZER_PATH = Path(os.environ["TOKENIZER_PATH"])
CHECKPOINT_PATH = Path(os.environ["CHECKPOINT_PATH"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--dataset", help="Path to dataset")
    parser.add_argument("--resume", help="Path to checkpoint to resume from")
    parser.add_argument(
        "--resume_from_last", action="store_true", help="Resume from last checkpoint"
    )
    parser.add_argument("--checkpoint_option", default="", help="Checkpoint option")
    parser.add_argument("--wandb_name", help="Wandb name")

    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    model_args, data_args, train_args = conf.model, conf.data, conf.train

    # get SLURM variables
    rank, local_rank, world_size, devices, num_nodes = read_slurm_env()
    print("RANK: ", rank)

    # seed everything
    pl.seed_everything(train_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    config = LOCOSTConfig(
        d_model=model_args.d_model,
        d_ff=model_args.d_ff,
        d_state=model_args.d_state,
        local_radius=model_args.local_radius,
        use_fast_fft_conv=model_args.use_fast_fft_conv,
        vocab_size=len(tokenizer),
        num_heads=model_args.num_heads,
        num_ssm_heads=model_args.num_ssm_heads,
        d_kv=model_args.d_model // model_args.num_heads,
        num_layers=model_args.num_layers,
        dropout_rate=model_args.dropout_rate,
        bidirectional=model_args.get("bidirectional", True),
        gating=model_args.get("gating", True),
        ssm_type="fullssm",
    )
    model = LOCOSTForConditionalGeneration(config)

    dataset = load_from_disk(args.dataset)
    dataset = dataset.with_format("torch")
    dataset = dataset.shuffle()
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    data_collator = DataCollator(
        tokenizer,
        model=model,
        padding="longest",
        max_length=model_args.max_length,
        truncate=True,
    )

    model_lit = LitLLMForConditionalGeneration(
        model,
        tokenizer=tokenizer,
        num_training_steps=train_args.training_steps,
        ratio_warmup=train_args.ratio_warmup,
        warmup_steps=train_args.warmup_steps,
        label_smoothing_factor=train_args.label_smoothing,
        lr=train_args.lr,
        scheduler=train_args.scheduler,
        optimizer_name=train_args.optimizer,
    )

    data_loader = DataLoader(
        train_dataset,
        batch_size=train_args.per_device_train_batch_size,
        shuffle=False,
        num_workers=data_args.num_workers,
        collate_fn=data_collator,
    )
    eval_data_loader = DataLoader(
        eval_dataset,
        batch_size=train_args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=data_args.num_workers,
        collate_fn=data_collator,
    )

    accumulate_grad_batches = train_args.effective_batch_size // (
        train_args.per_device_train_batch_size * world_size
    )
    checkpoint_save_dir = CHECKPOINT_PATH / (
        args.checkpoint_option + "_" + Path(args.config).stem
    )
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_save_dir,
        every_n_train_steps=train_args.save_steps,
    )

    if args.resume_from_last:
        checkpoint_path = max(checkpoint_save_dir.glob("*.ckpt"), key=os.path.getctime)
        current_step = int(re.search("step=(\d+)", checkpoint_path.name).group(1))
        max_steps = train_args.training_steps - current_step
    elif args.resume:
        checkpoint_path = Path(args.resume)
        current_step = int(re.search("step=(\d+)", checkpoint_path.name).group(1))
        max_steps = train_args.training_steps  # - current_step
    else:
        checkpoint_path = None
        max_steps = train_args.training_steps

    wandb_logger = WandbLogger(
        project=os.environ["WANDB_PROJECT"],
        name=args.wandb_name if args.wandb_name else None,
        config=OmegaConf.to_container(conf, resolve=True),
    )

    if (train_args.precision == "16") or (train_args.precision == "32"):
        precision = int(train_args.precision)
    else:
        precision = train_args.precision

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=None,
        precision=precision,
        devices=devices,
        num_nodes=num_nodes,
        log_every_n_steps=train_args.logging_steps,
        max_steps=max_steps,
        max_epochs=None,
        gradient_clip_val=1.0,
        strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False),
        val_check_interval=train_args.save_steps,
        callbacks=[
            checkpoint_callback,
            learning_rate_monitor,
        ],
    )

    print("Number of optimization steps:", trainer.estimated_stepping_batches)
    print("Number of warmup steps:", model_lit.num_warmup_steps)
    print("Number of remaining samples:", max_steps * train_args.effective_batch_size)

    torch.set_float32_matmul_precision("medium")
    trainer.fit(
        model_lit,
        train_dataloaders=data_loader,
        ckpt_path=checkpoint_path,
        val_dataloaders=eval_data_loader,
    )
