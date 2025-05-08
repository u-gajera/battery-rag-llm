import os
import json
from difflib import get_close_matches

CHUNKS_DIR = "data/processed/chunks"
QA_PATH = "data/qa/train.jsonl"
OUTPUT_PATH = "data/qa/train_fixed.jsonl"

# 1. Load all real chunk_ids
valid_chunk_ids = set()
for fn in os.listdir(CHUNKS_DIR):
    if fn.endswith(".jsonl"):
        with open(os.path.join(CHUNKS_DIR, fn), encoding="utf-8") as f:
            for line in f:
                try:
                    j = json.loads(line)
                    valid_chunk_ids.add(j["chunk_id"])
                except:
                    continue

print(f"✅ Found {len(valid_chunk_ids)} valid chunk_ids")

# 2. Read and fix QA records
fixed = []
skipped = 0

with open(QA_PATH, encoding="utf-8") as f:
    for i, line in enumerate(f):
        rec = json.loads(line)
        cid = rec["context_ids"][0]
        if cid in valid_chunk_ids:
            fixed.append(rec)
            continue
        # Try to find close match
        matches = get_close_matches(cid, valid_chunk_ids, n=1, cutoff=0.8)
        if matches:
            rec["context_ids"][0] = matches[0]
            fixed.append(rec)
        else:
            skipped += 1

print(f"✅ Fixed: {len(fixed)} | ❌ Skipped: {skipped}")

# 3. Write output
with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
    for rec in fixed:
        out.write(json.dumps(rec) + "\n")

print(f"📁 Saved corrected file to: {OUTPUT_PATH}")