# notebooks/02_evaluate_ragas.ipynb  (pseudocode)
from ragas.metrics import (
    answer_correctness, faithfulness, context_precision,
    context_recall, answer_relevance
)
from ragas import evaluate
import json, faiss, numpy as np

# 1) load eval set
eval_recs = [json.loads(l) for l in open("data/qa/eval.jsonl")]

# 2) build FAISS index from embeddings/*.npy  (flat L2 for demo)
vecs, ids = [], []
for fn in glob("data/processed/embeddings/*.npy"):
    arr = np.load(fn); vecs.append(arr)
    base = Path(fn).stem
    ids.extend([f"{base}_idx{i}" for i in range(arr.shape[0])])
index = faiss.IndexFlatIP(vecs[0].shape[1])
index.add(np.vstack(vecs))

# 3) retrieval + generation loop → RAGAS
rag_data = []
for rec in eval_recs:
    q_vec = retriever.encode(rec["question"])
    D,I = index.search(q_vec, k=5)
    retrieved = [ids[i] for i in I[0]]

    prompt = assemble_prompt(retrieved, rec["question"])
    answer = generator.generate(prompt)

    rag_data.append({"question": rec["question"],
                     "answer": answer,
                     "contexts": retrieved,
                     "ground_truth": rec["answer"]})

report = evaluate(rag_data,
                  metrics=[answer_correctness, faithfulness,
                           context_precision, context_recall,
                           answer_relevance])
print(report.to_pandas())
