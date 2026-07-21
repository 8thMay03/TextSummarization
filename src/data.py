from __future__ import annotations

from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from src.config import Config


def load_summarization_dataset(config: Config) -> DatasetDict:
    if config.dataset_config:
        return load_dataset(config.dataset_name, config.dataset_config)
    return load_dataset(config.dataset_name)


def preprocess_batch(
    examples: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
) -> dict[str, Any]:
    texts = examples[config.source_column]
    if config.prefix:
        texts = [config.prefix + text for text in texts]

    model_inputs = tokenizer(
        texts,
        max_length=config.max_source_length,
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples[config.target_column],
        max_length=config.max_target_length,
        truncation=True,
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def _maybe_subset(split: Dataset, max_samples: int | None, seed: int) -> Dataset:
    if max_samples is None:
        return split
    split = split.shuffle(seed=seed)
    return split.select(range(min(max_samples, len(split))))


def _tokenize_split(
    split: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
    max_samples: int | None,
) -> Dataset:
    split = _maybe_subset(split, max_samples, config.seed)
    return split.map(
        lambda batch: preprocess_batch(batch, tokenizer, config),
        batched=True,
        remove_columns=split.column_names,
    )


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: Config,
    *,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
    splits: list[str] | None = None,
) -> DatasetDict:
    split_limits = {
        "train": max_train_samples,
        "validation": max_eval_samples,
        "test": max_test_samples,
    }
    selected_splits = splits or [name for name, limit in split_limits.items() if limit is not None]

    # When all limits are None and splits not specified, tokenize every split.
    if not selected_splits:
        selected_splits = list(split_limits.keys())

    return DatasetDict(
        {
            name: _tokenize_split(dataset[name], tokenizer, config, split_limits[name])
            for name in selected_splits
            if name in dataset
        }
    )
