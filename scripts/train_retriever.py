# scripts/train_retriever.py
'''
python scripts/train_retriever.py \
  --chunks_dir data/processed/chunks \
  --train data/qa/train.jsonl \
  --base_model batterydata/batterybert-cased \
  --out_dir models/retriever_bbert_dpr
'''
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
import json, argparse, os

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

def main(args):
    # 1) build mapping chunk_id → text
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
                    except json.JSONDecodeError as je:
                        print(f"⚠️ JSON decode error in {fn} at line {i+1}: {je}")
                    except Exception as e:
                        print(f"⚠️ Unexpected error in {fn} at line {i+1}: {e}")
        except Exception as e:
            print(f"❌ Failed to open file {fn}: {e}")

    # 2) load training examples
    qa_pairs = load_qa(args.train)
    print(f"📊 Loaded {len(qa_pairs)} QA pairs from {args.train}")

    train_examples = []
    skipped = 0
    for q, cid in qa_pairs:
        if cid not in chunk_lookup:
            print(f"⚠️ Skipping example – chunk_id not found: {cid}")
            skipped += 1
            continue
        train_examples.append(InputExample(texts=[q, chunk_lookup[cid]]))
    print(f"✅ Final training examples: {len(train_examples)} (Skipped: {skipped})")

    if not train_examples:
        print("❌ No valid training data. Exiting.")
        return

    # 3) Training
    print("🚀 Starting retriever training...")
    loader = DataLoader(train_examples, batch_size=16, shuffle=True)
    model = SentenceTransformer(args.base_model)
    loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(train_objectives=[(loader, loss)],
              epochs=2,                # for quick test
              warmup_steps=5,
              output_path=args.out_dir)

    print(f"✅ Training complete. Model saved to: {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--base_model", default="batterydata/batterybert-cased")
    ap.add_argument("--out_dir", default="models/retriever_bbert_dpr")
    main(ap.parse_args())

