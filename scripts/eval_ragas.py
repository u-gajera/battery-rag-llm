#!/usr/bin/env python
'''
python eval_ragas.py \
  --eval_file data/qa/eval.jsonl \
  --model_dir models/generator_lora \
  --chunk_dir data/processed/chunks
'''
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from ragas import evaluate
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    context_precision,
    context_recall,
    answer_relevance
)
import argparse
from tqdm import tqdm

def load_eval_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto")
    return model, tokenizer

def assemble_prompt(context_chunks, question):
    context = "<context>\n" + "\n\n".join(context_chunks) + "\n</context>\n\n"
    return context + "### Question:\n" + question + "\n\n### Answer:\n"

def generate_answer(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_ragas_evaluation(eval_file, model_dir, chunk_dir):
    print("🔄 Loading evaluation records...")
    eval_recs = load_eval_jsonl(eval_file)

    print("📦 Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_dir)

    print("📚 Loading chunks...")
    chunk_lookup = {}
    for chunk_file in Path(chunk_dir).glob("*.jsonl"):
        with open(chunk_file, "r", encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                chunk_lookup[j["chunk_id"]] = j["text"]

    rag_data = []
    print("⚙️ Generating answers...")
    for rec in tqdm(eval_recs):
        chunk_texts = [chunk_lookup.get(cid, "") for cid in rec["context_ids"]]
        prompt = assemble_prompt(chunk_texts, rec["question"])
        answer = generate_answer(prompt, model, tokenizer)

        rag_data.append({
            "question": rec["question"],
            "answer": answer,
            "contexts": chunk_texts,
            "ground_truth": rec["answer"]
        })

    print("📊 Running RAGAS evaluation...")
    report = evaluate(
        rag_data,
        metrics=[
            answer_correctness,
            faithfulness,
            context_precision,
            context_recall,
            answer_relevance
        ]
    )
    print(report.to_pandas())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", required=True, help="Path to eval.jsonl")
    parser.add_argument("--model_dir", required=True, help="Path to fine-tuned generator model")
    parser.add_argument("--chunk_dir", required=True, help="Path to chunk JSONL files")

    args = parser.parse_args()
    run_ragas_evaluation(args.eval_file, args.model_dir, args.chunk_dir)
