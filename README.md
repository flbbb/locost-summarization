# LOCOST

This repo contains the code used to pretrain and finetune LOCOST.

The scripts about state-space models are adapted from the official [H3 repository](https://github.com/HazyResearch/H3).

## Setup
Install both packages in the `csrc/` folder:
```bash
cd csrc
cd fftconv
pip install ./
cd ../cauchy
pip install ./
```

## Data

We expect the datasets to be tokenized with the base LongT5 tokenizer. This formatting can be done with the script `preprocess_data.py`.

## Env

These scripts rely on a `.env` file, and is used through the [python-dotenv](https://pypi.org/project/python-dotenv/) package. Make sure to define here:

- `DATASET_PATH`, the base folder where are stored the dataset.
- `TOKENIZER_PATH`, the path to the model tokenizer (we used the LongT5 tokenizer).
- `CHECKPOINT_PATH` to save the models checkpoint during training.

# Pretraining

The pretraining is ran with PytorchLightning and tracked with `wandb`.

```bash
TRANSFORMERS_NO_ADVISORY_WARNINGS="true" python pretrain_script.py --dataset path/to/pretraining/dataset --config configs/pretraining/locost.yaml --wandb_name locost-pretraining
```
