import pytorch_lightning as pl
from transformers.trainer_pt_utils import LabelSmoother

from utils import decode_batch_labels, eval_rouge, get_optimizer, get_scheduler


class LitLLMForConditionalGeneration(pl.LightningModule):
    def __init__(
        self,
        model,
        tokenizer,
        num_training_steps=None,
        lr=5e-4,
        lr_end=5e-6,
        repetition_penalty=2.5,
        max_target_tokens=80,
        num_beams=4,
        ratio_warmup=0.1,
        batch_size=16,
        label_smoothing_factor=0.0,
        scheduler="cosine",
        warmup_steps=None,
        optimizer_name="adamw",
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.max_target_tokens = max_target_tokens
        self.repetition_penalty = repetition_penalty
        self.num_beams = num_beams
        self.tokenizer = tokenizer
        self.num_training_steps = num_training_steps

        if warmup_steps is not None:
            self.num_warmup_steps = warmup_steps
        else:
            assert ratio_warmup is not None
            self.num_warmup_steps = int(ratio_warmup * num_training_steps)
        self.lr_end = lr_end
        self.batch_size = batch_size
        self.label_smoothing_factor = label_smoothing_factor
        self.scheduler = scheduler
        self.optimizer_name = optimizer_name

        if self.label_smoothing_factor > 0.0:
            self.label_smoother = LabelSmoother(epsilon=self.label_smoothing_factor)
        else:
            self.label_smoother = None
        self.save_hyperparameters({"config": self.model.config})

    def training_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        if self.label_smoother is not None:
            loss = self.label_smoother(out, batch["labels"])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def forward(
        self,
        input_ids=None,
        labels=None,
        decoder_input_ids=None,
        encoder_outputs=None,
        attention_mask=None,
    ):
        return self.model(
            input_ids=input_ids,
            labels=labels,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
        )

    def configure_optimizers(self):
        optimizer_cls = get_optimizer(self.optimizer_name)

        optimizer = optimizer_cls(
            [p for p in self.parameters()],
            lr=self.lr,
        )
        lr_scheduler = get_scheduler(
            scheduler_name=self.scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps,
            lr_end=self.lr_end,
        )

        scheduler = {
            "scheduler": lr_scheduler,
            "name": "lr",
            "interval": "step",
        }
        return [optimizer], [scheduler]


class LitLLMForSummarization(LitLLMForConditionalGeneration):
    def __init__(
        self,
        repetition_penalty=1.0,
        length_penalty=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.repetition_penalty = repetition_penalty
        self.length_penalty = length_penalty

    def validation_step(self, batch, batch_idx):
        out = self(**batch)
        loss = out.loss
        self.log("val_loss", loss, sync_dist=True)

        top_beam_ids = self.model.generate(
            inputs=batch["input_ids"],
            max_new_tokens=512,
            attention_mask=batch["attention_mask"],
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
        )
        labels = batch["labels"]
        labels[labels == -100] = self.tokenizer.pad_token_id

        target = decode_batch_labels(self.tokenizer, labels)
        prediction = decode_batch_labels(self.tokenizer, top_beam_ids)
        rouge_results = eval_rouge(
            list_predictions=prediction,
            list_targets=target,
            n_print=0,
        )

        self.log_dict(rouge_results, sync_dist=True)
        return loss
