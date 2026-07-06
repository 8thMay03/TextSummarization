from __future__ import annotations

import argparse

import evaluate
import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.config import DEFAULT_CONFIG
from src.data import load_summarization_dataset, tokenize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned summarization model.")
    parser.add_argument("--model-path", default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--max-test-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)
    raw_dataset = load_summarization_dataset(config)
    tokenized_dataset = tokenize_dataset(
        raw_dataset,
        tokenizer,
        config,
        max_test_samples=args.max_test_samples,
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge.compute(predictions=decoded_predictions, references=decoded_labels, use_stemmer=True)
        return {key: round(value * 100, 4) for key, value in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir="outputs/evaluation",
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
