from __future__ import annotations

import argparse

import evaluate
import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from src.config import DEFAULT_CONFIG, SummarizationConfig
from src.data import load_summarization_dataset, tokenize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ViT5 for Vietnamese summarization.")
    parser.add_argument("--model-name", default=DEFAULT_CONFIG.model_name)
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int, default=20_000)
    parser.add_argument("--max-eval-samples", type=int, default=2_000)
    parser.add_argument("--full-dataset", action="store_true", help="Train on the entire training split.")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SummarizationConfig:
    return SummarizationConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        seed=args.seed,
    )


def compute_rouge_metrics(tokenizer, eval_pred):
    rouge = evaluate.load("rouge")
    predictions, labels = eval_pred
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(
        predictions=decoded_predictions,
        references=decoded_labels,
        use_stemmer=False,
    )
    return {key: round(value * 100, 4) for key, value in result.items()}


def main() -> None:
    args = parse_args()
    config = build_config(args)

    max_train_samples = None if args.full_dataset else args.max_train_samples

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    raw_dataset = load_summarization_dataset(config)
    tokenized_dataset = tokenize_dataset(
        raw_dataset,
        tokenizer,
        config,
        max_train_samples=max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.06,
        fp16=args.fp16 and torch.cuda.is_available(),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=config.max_target_length,
        generation_num_beams=config.generation_num_beams,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        seed=config.seed,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=lambda eval_pred: compute_rouge_metrics(tokenizer, eval_pred),
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


if __name__ == "__main__":
    main()
