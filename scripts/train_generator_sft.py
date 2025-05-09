
# pip install bitsandbytes trl peft accelerate

"""
python scripts/train_generator_sft.py \
  --train data/qa/train.jsonl \
  --chunks_dir data/processed/chunks \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --out_dir models/generator_lora
"""

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json, argparse

def load_chunk_lookup(chunks_dir="/data/processed/chunks"):
    lookup = {}
    import os
    for fn in os.listdir(chunks_dir):
        if not fn.endswith(".jsonl"): continue
        with open(os.path.join(chunks_dir, fn), encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                lookup[rec["chunk_id"]] = rec["text"]
    return lookup

def stream_sft_records(path, chunk_lookup):
    for line in open(path, encoding="utf-8"):
        j = json.loads(line)
        prompt = ("<context>\n" +
                  "\n\n".join(chunk_lookup.get(cid, f"[MISSING {cid}]") for cid in j["context_ids"]) +
                  "\n</context>\n\n" +
                  "### Question:\n" + j["question"] +
                  "\n\n### Answer:\n")
        yield {"prompt": prompt, "answer": j["answer"]}

def main(args):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype="bfloat16",
        bnb_4bit_quant_type="nf4"
    )

    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            device_map="auto",
            quantization_config=bnb_config
    )

        # 🔧 Fix for 4-bit + PEFT training
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora)

    chunk_lookup = load_chunk_lookup(args.chunks_dir)
    ds = list(stream_sft_records(args.train, chunk_lookup))

    def tokenize(example):
        encoded = tok(
            example["prompt"] + example["answer"],
            truncation=True,
            max_length=1024,
            padding="max_length"
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    tokenized = [tokenize(ex) for ex in ds]
    train_dataset = Dataset.from_list(tokenized)

    collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=False)

    train_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        remove_unused_columns=False
    )

    model.gradient_checkpointing_enable()
    model.train()

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=train_args,
        data_collator=collator,
        tokenizer=tok
    )

    trainer.train()
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--chunks_dir", default="data/processed/chunks")
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--out_dir", default="models/generator_lora")
    main(ap.parse_args())
