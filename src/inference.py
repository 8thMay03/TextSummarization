from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import DEFAULT_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a summary with T5.")
    parser.add_argument("--model-path", default=DEFAULT_CONFIG.output_dir)
    parser.add_argument("--text", required=True)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--min-length", type=int, default=30)
    parser.add_argument("--num-beams", type=int, default=4)
    return parser.parse_args()


def summarize(
    text: str,
    model_path: str = DEFAULT_CONFIG.output_dir,
    max_length: int = 128,
    min_length: int = 30,
    num_beams: int = 4,
) -> str:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(
        DEFAULT_CONFIG.prefix + text,
        return_tensors="pt",
        max_length=DEFAULT_CONFIG.max_source_length,
        truncation=True,
    ).to(device)

    summary_ids = model.generate(
        **inputs,
        max_length=max_length,
        min_length=min_length,
        num_beams=num_beams,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def main() -> None:
    args = parse_args()
    print(
        summarize(
            text=args.text,
            model_path=args.model_path,
            max_length=args.max_length,
            min_length=args.min_length,
            num_beams=args.num_beams,
        )
    )


if __name__ == "__main__":
    main()
