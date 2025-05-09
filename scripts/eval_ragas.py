#!/usr/bin/env python
"""
Evaluate a Retrieval-Augmented Generation (RAG) pipeline using RAGAS metrics
with an open-source Hugging Face model (no ChatGPT API).

Usage:
  python scripts/eval_ragas.py \
    --eval_file data/qa/eval.jsonl \
    --model_dir models/generator_lora \
    --chunk_dir data/processed/chunks
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    answer_correctness,
    faithfulness,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper


def load_eval_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        torch_dtype="auto"
    )
    return model, tokenizer


def assemble_prompt(context_chunks, question):
    ctx = "<context>\n" + "\n\n".join(context_chunks) + "\n</context>\n\n"
    return f"{ctx}### Question:\n{question}\n\n### Answer:\n"


def generate_answer(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def run_rag_evaluation(eval_file, model_dir, chunk_dir):
    # 1) Load eval records
    print("🔄 Loading evaluation records...")
    eval_recs = load_eval_jsonl(eval_file)

    # 2) Load generator model & tokenizer
    print("📦 Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_dir)

    # 3) Wrap with HuggingFacePipeline → Langchain → RAGAS
    print("🤖 Wrapping model in HuggingFacePipeline + LangchainLLMWrapper...")
    hf_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"max_new_tokens": 256, "temperature": 0.0}
    )
    lc_llm = HuggingFacePipeline(pipeline=hf_pipe)
    rag_llm = LangchainLLMWrapper(lc_llm)

    # 4) Set up open‐source embeddings
    print("🔧 Setting up open-source embeddings...")
    hf_emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 5) Load context chunks
    print("📚 Loading chunks...")
    chunk_lookup = {}
    for path in Path(chunk_dir).glob("*.jsonl"):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                chunk_lookup[obj["chunk_id"]] = obj["text"]

    # 6) Generate answers (with correct column names)
    print("⚙️ Generating answers...")
    rag_data = []
    for rec in tqdm(eval_recs, desc="Records"):
        contexts = [ chunk_lookup[cid] for cid in rec["context_ids"] ]
        prompt   = assemble_prompt(contexts, rec["question"])
        ans      = generate_answer(prompt, model, tokenizer)

        rag_data.append({
            "user_input":         rec["question"],
            "response":           ans,
            "reference":          rec["answer"],
            "contexts":           contexts,
            "retrieved_contexts": contexts,
        })

    # 7) Wrap into an EvaluationDataset
    print("🔄 Wrapping predictions in EvaluationDataset...")
    eval_dataset = EvaluationDataset.from_list(rag_data)

    # 8) Run RAGAS evaluation
    print("📊 Running RAGAS evaluation...")
    report = evaluate(
        eval_dataset,
        metrics=[
            answer_correctness,
            faithfulness,
            context_precision,
            context_recall
        ],
        llm=rag_llm,
        embeddings=hf_emb,
        show_progress=True
    )

    # 9) Print results
    df = report.to_pandas()
    print("\n=== RAG Evaluation Report ===")
    try:
        # prettier Markdown table if tabulate is installed
        print(df.to_markdown(index=False))
    except ImportError:
        # fallback to plain text
        print(df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_file", required=True,
        help="Path to your eval.jsonl (with fields question, answer, context_ids)"
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Path to your fine-tuned generator model directory"
    )
    parser.add_argument(
        "--chunk_dir", required=True,
        help="Directory containing your chunk JSONL files"
    )
    args = parser.parse_args()
    run_rag_evaluation(args.eval_file, args.model_dir, args.chunk_dir)
 