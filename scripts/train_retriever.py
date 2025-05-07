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
    for line in open(path, encoding='utf-8'):
        rec = json.loads(line)
        yield rec['question'], rec['context_ids'][0]   # first chunk as positive

def main(args):
    # 1) build mapping chunk_id → text
    chunk_lookup = {}
    for fn in os.listdir(args.chunks_dir):
        for line in open(os.path.join(args.chunks_dir, fn), encoding='utf-8'):
            j = json.loads(line); chunk_lookup[j['chunk_id']] = j['text']

    train_examples = [InputExample(texts=[q, chunk_lookup[cid]])
                      for q, cid in load_qa(args.train)]
    loader = DataLoader(train_examples, batch_size=16, shuffle=True)

    model = SentenceTransformer(args.base_model)   # e.g. batterydata/batterybert-cased
    loss  = losses.MultipleNegativesRankingLoss(model)

    model.fit(train_objectives=[(loader, loss)],
              epochs=3,
              warmup_steps=100,
              output_path=args.out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks_dir", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--base_model", default="batterydata/batterybert-cased")
    ap.add_argument("--out_dir", default="models/retriever_bbert_dpr")
    main(ap.parse_args())
