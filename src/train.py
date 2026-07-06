from __future__ import annotations

import argparse

import evaluate
import numpy as np
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.config import DEFAULT_CONFIG
from src.data import load_summarization_dataset, tokenize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune T5-small for English summarization.")
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-train-samples", type=int, default=1000)
    parser.add_argument("--max-eval-samples", type=int, default=200)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_CONFIG

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    raw_dataset = load_summarization_dataset(config)
    tokenized_dataset = tokenize_dataset(
        raw_dataset,
        tokenizer,
        config,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    rouge = evaluate.load("rouge")

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        result = rouge.compute(
            predictions=decoded_predictions,
            references=decoded_labels,
            use_stemmer=True,
        )
        return {key: round(value * 100, 4) for key, value in result.items()}

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        fp16=args.fp16,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
