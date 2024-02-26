import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from lightning_datamodules import SummaryDataModule
from models import LOCOSTForConditionalGeneration
from models_config import LOCOSTConfig
from models_lightning import LitLLMForSummarization
from utils import read_slurm_env

load_dotenv()
TOKENIZER_PATH = Path(os.environ["TOKENIZER_PATH"])
CHECKPOINT_PATH = Path(os.environ["CHECKPOINT_PATH"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="Path to checkpoint to resume from.")
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--checkpoint_option", default="", help="Checkpoint option.")
    parser.add_argument("--wandb_name", help="Wandb name.")
    parser.add_argument("--model_path", help="Pretrained model path.")
    parser.add_argument("--data_path", help="Path to finetuning dataset.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--model_name", help="Model name.", default="locost")

    args = parser.parse_args()

    conf = OmegaConf.load(args.config)
    model_args, train_args = conf.model, conf.train

    # get SLURM variables
    rank, local_rank, world_size, devices, num_nodes = read_slurm_env()
    print("RANK: ", rank)

    # seed everything
    pl.seed_everything(train_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model_config = LOCOSTConfig.from_pretrained(args.model_path)
    model_config.dropout_rate = model_args.dropout_rate
    model = LOCOSTForConditionalGeneration(model_config)

    checkpoint = torch.load(
        Path(args.model_path) / "pytorch_model.bin", map_location="cpu"
    )
    model.load_state_dict(checkpoint)

    print("Model loaded.")
    data_module = SummaryDataModule(
        data_path=args.data_path,
        train_batch_size=train_args.per_device_train_batch_size,
        eval_batch_size=train_args.per_device_eval_batch_size,
        tokenizer=tokenizer,
        max_length=model_args.max_length,
        num_workers=args.num_workers,
    )
    num_training_steps = train_args.get("num_training_steps", None)

    model_lit = LitLLMForSummarization(
        model=model,
        tokenizer=tokenizer,
        warmup_steps=train_args.warmup_steps,
        label_smoothing_factor=train_args.label_smoothing,
        num_training_steps=num_training_steps,
        lr=train_args.lr,
        scheduler=train_args.scheduler,
        optimizer_name=train_args.optimizer,
    )

    accumulate_grad_batches = train_args.effective_batch_size // (
        train_args.per_device_train_batch_size * world_size
    )

    model_name = f"{args.model_name}_{Path(args.data_path).stem}"
    learning_rate_monitor = LearningRateMonitor(logging_interval="step")
    top_k_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH
        / (
            "top_k" + args.checkpoint_option + model_name + "_" + Path(args.config).stem
        ),
        every_n_epochs=5,
        save_top_k=1,
        monitor="mean-rouge",
        mode="max",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_PATH
        / (args.checkpoint_option + model_name + "_" + Path(args.config).stem),
        every_n_epochs=5,
    )

    wandb_logger = WandbLogger(
        project=os.environ["WANDB_PROJECT"],
        name=args.wandb_name if args.wandb_name else None,
        config=OmegaConf.to_container(conf, resolve=True),
    )

    if (train_args.precision == "16") or (train_args.precision == "32"):
        precision = int(train_args.precision)
    else:
        precision = train_args.precision

    if hasattr(train_args, "grad_clip"):
        grad_clip = train_args.grad_clip
    else:
        grad_clip = 1.0

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        accumulate_grad_batches=accumulate_grad_batches,
        check_val_every_n_epoch=train_args.check_val_epoch,
        precision=precision,
        devices=devices,
        num_nodes=num_nodes,
        log_every_n_steps=train_args.logging_steps,
        max_epochs=train_args.n_epochs,
        gradient_clip_val=grad_clip,
        strategy=pl.strategies.ddp.DDPStrategy(find_unused_parameters=False),
        val_check_interval=None,
        callbacks=[
            checkpoint_callback,
            top_k_callback,
            learning_rate_monitor,
        ],
    )

    print("Number of optimization steps:", trainer.estimated_stepping_batches)
    print("Number of warmup steps:", model_lit.num_warmup_steps)

    torch.set_float32_matmul_precision("medium")
    trainer.fit(
        model_lit,
        datamodule=data_module,
        ckpt_path=args.resume,
    )
