from __future__ import annotations

from typing import Any

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

from src.config import SummarizationConfig


def load_summarization_dataset(config: SummarizationConfig) -> DatasetDict:
    return load_dataset(config.dataset_name, config.dataset_config)


def preprocess_batch(
    examples: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    config: SummarizationConfig,
) -> dict[str, Any]:
    inputs = [config.prefix + text for text in examples[config.source_column]]
    model_inputs = tokenizer(
        inputs,
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


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    config: SummarizationConfig,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
    max_test_samples: int | None = None,
) -> DatasetDict:
    if max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(max_train_samples, len(dataset["train"]))))
    if max_eval_samples:
        dataset["validation"] = dataset["validation"].select(
            range(min(max_eval_samples, len(dataset["validation"])))
        )
    if max_test_samples:
        dataset["test"] = dataset["test"].select(range(min(max_test_samples, len(dataset["test"]))))

    return dataset.map(
        lambda batch: preprocess_batch(batch, tokenizer, config),
        batched=True,
        remove_columns=dataset["train"].column_names,
    )
