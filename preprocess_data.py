import argparse
import os

from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
DATA_PATH = os.environ["DATA_PATH"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", help="Dataset name")
    parser.add_argument("--dataset_subset", help="Dataset subset")
    parser.add_argument("--data_column", help="Data column")
    parser.add_argument("--text_column", help="Text column")
    parser.add_argument("--max_source_length", type=int, help="Max source length")
    parser.add_argument("--max_target_length", type=int, help="Max target length")
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=30,
        help="Preprocessing num workers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Preprocessing batch size",
    )

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google/long-t5-local-base")

    kwargs = {}
    if args.dataset_subset is not None:
        kwargs["name"] = args.dataset_subset
    dataset = load_dataset(args.dataset_name, **kwargs)
    data_column, text_column = args.data_column, args.text_column

    def preprocess_function(examples):
        inputs = examples[data_column]
        targets = examples[text_column]
        model_inputs = tokenizer(
            inputs,
            max_length=args.max_source_length,
            truncation=True,
        )

        if None in targets:
            return model_inputs
        else:
            labels = tokenizer(
                text_target=targets,
                max_length=args.max_target_length,
                truncation=True,
            )
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

    column_names = dataset["train"].column_names
    dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=args.batch_size,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Running tokenizer on dataset",
    )
    if args.dataset_subset is not None:
        path = f"{DATA_PATH}{args.dataset_subset.split('/')[-1]}_{args.max_source_length}_{args.max_target_length}/"
    else:
        path = f"{DATA_PATH}{args.dataset_name.split('/')[-1]}_{args.max_source_length}_{args.max_target_length}/"
    dataset.save_to_disk(path)
