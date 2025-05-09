# scripts/train_retriever.py
'''
python scripts/train_retriever.py \
  --chunks_dir data/processed/chunks \
  --train data/qa/train.jsonl \
  --base_model batterydata/batterybert-cased \
  --out_dir models/retriever_bbert_dpr
'''
from sentence_transformers import SentenceTransformer, losses, InputExample, models, util
from torch.utils.data import DataLoader
import json, argparse, os
from tqdm import tqdm

def load_qa(path):
    examples = []
    for i, line in enumerate(open(path, encoding='utf-8')):
        try:
            rec = json.loads(line)
            question = rec['question']
            context_id = rec['context_ids'][0]
            examples.append((question, context_id))
        except Exception as e:
            print(f"⚠️ Error in QA line {i+1}: {e}")
    return examples

def load_sentence_transformer(base_model):
    if os.path.isdir(base_model) or base_model.startswith("sentence-transformers/"):
        print(f"✅ Loading SentenceTransformer from: {base_model}")
        return SentenceTransformer(base_model)
    try:
        print(f"🔧 Wrapping HuggingFace model: {base_model}")
        word_embedding_model = models.Transformer(base_model)
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode_mean_tokens=True
        )
        return SentenceTransformer(modules=[word_embedding_model, pooling_model])
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load model: {base_model}\n{e}")

def mine_hard_negatives(query_model, queries, context_ids, chunk_lookup, k=2):
    print("🔍 Mining hard negatives...")
    chunk_ids = list(chunk_lookup.keys())
    chunk_texts = [chunk_lookup[cid] for cid in chunk_ids]
    chunk_embeddings = query_model.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=True)

    hard_negatives = {}
    for i, query in tqdm(enumerate(queries), total=len(queries)):
        q_embedding = query_model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(q_embedding, chunk_embeddings, top_k=k+5)[0]
        selected = []
        for hit in hits:
            candidate_id = chunk_ids[hit['corpus_id']]
            if candidate_id != context_ids[i] and candidate_id not in selected:
                selected.append(candidate_id)
            if len(selected) == k:
                break
        hard_negatives[i] = selected
    return hard_negatives

def main(args):
    # 1) Build chunk_id → text
    chunk_lookup = {}
    for fn in os.listdir(args.chunks_dir):
        if not fn.endswith(".jsonl"):
            continue
        path = os.path.join(args.chunks_dir, fn)
        try:
            with open(path, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        j = json.loads(line)
                        chunk_lookup[j['chunk_id']] = j['text']
                    except Exception as e:
                        print(f"⚠️ JSON decode error in {fn} line {i+1}: {e}")
        except Exception as e:
            print(f"❌ Failed to open file {fn}: {e}")

    # 2) Load QA
    qa_pairs = load_qa(args.train)
    print(f"📊 Loaded {len(qa_pairs)} QA pairs from {args.train}")

    questions, context_ids = zip(*qa_pairs)

    # 3) Mine hard negatives using MiniLM or similar
    miner_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    hard_negs = mine_hard_negatives(miner_model, questions, context_ids, chunk_lookup, k=2)

    # 4) Prepare InputExamples
    train_examples = []
    skipped = 0
    for i, (q, cid) in enumerate(zip(questions, context_ids)):
        if cid not in chunk_lookup:
            skipped += 1
            continue
        positives = chunk_lookup[cid]
        negatives = [chunk_lookup[nid] for nid in hard_negs.get(i, []) if nid in chunk_lookup]
        if not negatives:
            skipped += 1
            continue
        for neg in negatives:
            train_examples.append(InputExample(texts=[q, positives, neg]))

    print(f"✅ Final training triplets: {len(train_examples)} (Skipped: {skipped})")

    if not train_examples:
        print("❌ No valid training data. Exiting.")
        return

    # 5) Load model to fine-tune
    model = load_sentence_transformer(args.base_model)

    # 6) Train
    print("🚀 Starting retriever training with hard negatives...")
    loader = DataLoader(train_examples, batch_size=16, shuffle=True)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(train_objectives=[(loader, loss)],
              epochs=3,
              warmup_steps=100,
              output_path=args.out_dir)

    print(f"✅ Training complete. Model saved to: {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--out_dir", required=True)
    main(ap.parse_args())
