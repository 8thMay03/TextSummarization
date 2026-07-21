from __future__ import annotations

import argparse
import sys

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.config import DEFAULT_CONFIG, Config
from src.data import load_summarization_dataset, tokenize_dataset
from src.train import compute_rouge_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned ViT5 summarization model.")
    parser.add_argument("--model-path", default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--dataset-name", default=DEFAULT_CONFIG.dataset_name)
    parser.add_argument("--max-test-samples", type=int, default=DEFAULT_CONFIG.test_sample_size)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    return Config(
        dataset_name=args.dataset_name,
        output_dir=args.model_path,
    )


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    config = build_config(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    raw_dataset = load_summarization_dataset(config)
    tokenized_dataset = tokenize_dataset(
        raw_dataset,
        tokenizer,
        config,
        max_test_samples=args.max_test_samples,
        splits=["test"],
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir="outputs/evaluation",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        generation_num_beams=config.generation_num_beams,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset["test"],
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=lambda eval_pred: compute_rouge_metrics(tokenizer, eval_pred),
    )

    metrics = trainer.evaluate()
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
