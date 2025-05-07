# pip install bitsandbytes trl peft accelerate
'''
python scripts/train_generator_sft.py \
  --train data/qa/train.jsonl \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --out_dir models/generator_lora
'''
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
import json, random, argparse, os

def stream_sft_records(path):
    for line in open(path, encoding='utf-8'):
        j = json.loads(line)
        prompt  = ("<context>\n" +
                   "\n\n".join(j["context_ids"]) +  # retrieved later; keep placeholder
                   "\n</context>\n\n" +
                   "### Question:\n" + j["question"] +
                   "\n\n### Answer:\n")
        yield {"prompt": prompt, "answer": j["answer"]}

def main(args):
    tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 load_in_4bit=True,
                                                 device_map="auto")
    lora = LoraConfig(r=8, alpha=16, target_modules=["q_proj","v_proj"])
    model = get_peft_model(model, lora)

    ds = list(stream_sft_records(args.train))
    def tokenize(example):
        return tok(example["prompt"] + example["answer"],
                   truncation=True, max_length=1024)

    train_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        num_train_epochs=2,
        learning_rate=2e-4,
        fp16=True,
    )
    model.train()
    model.gradient_checkpointing_enable()

    from trl import SFTTrainer
    trainer = SFTTrainer(model=model,
                         train_dataset=[tokenize(ex) for ex in ds],
                         args=train_args)
    trainer.train()
    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--base_model", default="mistralai/Mistral-7B-Instruct-v0.2")
    ap.add_argument("--out_dir", default="models/generator_lora")
    main(ap.parse_args())
