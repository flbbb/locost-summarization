import pytorch_lightning as pl
from datasets import load_from_disk
from torch.utils.data import DataLoader

from utils import DataCollator


class SummaryDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        train_batch_size,
        tokenizer,
        eval_batch_size=None,
        max_length=4096,
        eval_size=500,
        num_workers=8,
        train_size=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.eval_batch_size = (
            train_batch_size if eval_batch_size is None else eval_batch_size
        )
        self.max_length = max_length
        self.eval_size = eval_size
        self.num_workers = num_workers
        self.train_size = train_size

    def setup(self, stage):
        if stage == "fit":
            dataset = load_from_disk(self.data_path)
            dataset = dataset.with_format("torch")
            self.train_dataset = dataset["train"]
            if self.train_size is not None:
                self.train_dataset = self.train_dataset.select(range(self.train_size))
            self.eval_dataset = dataset["validation"].select(range(self.eval_size))
            self.data_collator = DataCollator(
                self.tokenizer,
                padding="longest",
                max_length=self.max_length,
                truncate=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            batch_size=self.train_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
        )
