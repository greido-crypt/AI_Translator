import argparse

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description="Translate one EN sentence to RU")
    parser.add_argument("--text", required=True, help="Source sentence in English")
    parser.add_argument(
        "--model_path",
        default="Helsinki-NLP/opus-mt-en-ru",
        help="Path to fine-tuned model directory or HF model name",
    )
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Max generated tokens")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    translation = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    print(f"Source: {args.text}")
    print(f"Translation: {translation}")


if __name__ == "__main__":
    main()
