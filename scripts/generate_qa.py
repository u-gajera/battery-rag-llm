#!/usr/bin/env python3
"""
Generate Q-A pairs from pre-chunked JSONL files.
  • mines candidate sentences
  • prompts a local LLM (e.g. mistralai/Mistral-7B-Instruct-v0.2)
  • filters with a cross-encoder (BatteryBERT-QA) à la SciQAG
  • writes train / eval JSONL
Usage:
  python scripts/generate_qa.py \
         --chunks_dir data/processed/chunks \
         --llm mistralai/Mistral-7B-Instruct-v0.2 \
         --cross_enc batterydata/batterybert-cross-encoder \
         --out_train data/qa/train.jsonl \
         --out_eval  data/qa/eval.jsonl
"""
import argparse, glob, json, os, random, re, uuid
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

SENT_RE = re.compile(r"(?<=[\.\!\?])\s+")

def pick_sentences(text, k=2):
    sents = SENT_RE.split(text)
    scored = []
    for s in sents:
        tok_len = len(s.split())
        if 5 < tok_len <= 40 and re.search(r"\d", s):
            score = 1
        elif "because" in s or "therefore" in s:
            score = 0.8
        else:
            continue
        scored.append((score, s.strip()))
    random.shuffle(scored)
    scored.sort(reverse=True)
    return [s for _, s in scored[:k]]

def draft_qa(llm_pipe, sent):
    prompt = (f"You are a battery-chemistry tutor. "
              f"Use only the given text to draft ONE question that a grad-student "
              f"might ask and its short factual answer.\n\n"
              f"### Text:\n{sent}\n\n### Question:")
    resp = llm_pipe(prompt, max_new_tokens=128, temperature=0.7)[0]['generated_text']
    # split back into Q / A
    if "### Answer:" in resp:
        q_part, a_part = resp.split("### Answer:", 1)
        question = q_part.split("### Question:")[-1].strip()
        answer = a_part.strip().split("\n")[0]
        return question, answer
    return None, None

def main(args):
    # load models
    print("Loading models …")
    llm_tok  = AutoTokenizer.from_pretrained(args.llm)
    llm_model= AutoModelForCausalLM.from_pretrained(
                    args.llm, torch_dtype="auto", device_map="auto")
    llm_pipe = pipeline("text-generation", model=llm_model, tokenizer=llm_tok)
    ce_model = CrossEncoder(args.cross_enc)
    
    train_out = open(args.out_train, "w", encoding="utf-8")
    eval_out  = open(args.out_eval,  "w", encoding="utf-8")

    for chunk_file in glob.glob(os.path.join(args.chunks_dir, "*.jsonl")):
        with open(chunk_file, encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh]

        for rec in lines:
            for sent in pick_sentences(rec["text"]):
                q, a = draft_qa(llm_pipe, sent)
                if not q:
                    continue

                # quality score via cross-encoder  (query, chunk)
                score = float(ce_model.predict([(q, rec["text"])]))
                if score < 0.8:
                    continue   # drop weak pairs

                out = {
                    "id": f"qa_{uuid.uuid4().hex[:8]}",
                    "question": q,
                    "answer": a,
                    "context_ids": [rec["chunk_id"]],
                    "doc_refs": [f"{rec['doc_id']}.pdf"],
                    "meta": {}
                }
                target = train_out if random.random() < 0.8 else eval_out
                target.write(json.dumps(out, ensure_ascii=False) + "\n")

    train_out.close(); eval_out.close()
    print("Done.  QA files written.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--cross_enc", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_eval", required=True)
    args = ap.parse_args()
    main(args)
