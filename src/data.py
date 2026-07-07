from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from src.config import SummarizationConfig


def load_summarization_dataset(config: SummarizationConfig) -> DatasetDict:
    if config.dataset_config:
        return load_dataset(config.dataset_name, config.dataset_config)
    return load_dataset(config.dataset_name)


def preprocess_batch(
    examples: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    config: SummarizationConfig,
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


def _maybe_subset(split, max_samples: int | None, seed: int):
    if not max_samples:
        return split
    split = split.shuffle(seed=seed)
    return split.select(range(min(max_samples, len(split))))


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: SummarizationConfig,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DatasetDict:
    tokenized = DatasetDict(
        {
            split: _maybe_subset(dataset[split], max_samples, config.seed)
            for split, max_samples in (
                ("train", max_train_samples),
                ("validation", max_eval_samples),
                ("test", max_test_samples),
            )
            if split in dataset
        }
    )

    return tokenized.map(
        lambda batch: preprocess_batch(batch, tokenizer, config),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
