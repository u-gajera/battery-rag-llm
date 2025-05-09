# RAG + LLM Model for Battery Chemistry - README

This repository outlines the steps to build and fine-tune a Retrieval-Augmented Generation (RAG) model for battery chemistry research. By integrating a retrieval system with a large language model (LLM), this pipeline allows you to answer domain-specific questions using a dataset of scientific papers, such as those found in the battery materials domain.

This guide takes you through everything from preparing datasets to fine-tuning the model and evaluating the results.

## 1. Setup and Preparation

### 1.1 Folder Structure

Ensure that your project directory is structured as follows:

```
battery-rag-llm/
├── data/
│   ├── processed/
│   │   ├── chunks/
│   │   └── embeddings/
│   └── qa/
│       ├── train.jsonl
│       └── eval.jsonl
├── models/
├── scripts/
│   ├── generate_qa.py
│   ├── train_retriever.py
│   └── train_generator_sft.py
└── notebooks/
    └── 02_evaluate_ragas.ipynb
```

### 1.2 Install Dependencies

To run the pipeline, make sure to install the following dependencies:

```bash
pip install torch transformers datasets sentence-transformers bitsandbytes peft trl accelerate ragas
```

### 1.3 Download and Preprocess Papers

1. **Download Papers**: Collect a corpus of research papers (e.g., from arXiv or other journals). You can use tools like `arxivpy` to bulk-download papers.

2. **Preprocess Papers**: Process the papers into text chunks. Each chunk will represent a passage from the paper and should be stored as a JSON file in the `chunks/` directory.

3. **Chunking and Embedding**: Generate the embeddings of the paper chunks. These embeddings will be used for the retrieval model.

   The script `build_embeddings.py` can be used to convert raw paper text into chunked embeddings.

### 1.4 Integrate BatteryBERT

For better results with battery-related terminology, you can use **BatteryBERT** or **SchuBERT** as a specialized encoder. Replace the `base_model` with `batterydata/batterybert-cased` in the script, ensuring the tokenizer understands domain-specific terms.

---

## 2. Data Preparation

### 2.1 Create Question-Answer Pairs

To fine-tune the RAG model, you need to create a question-answer dataset. This dataset will be stored as two JSONL files: `train.jsonl` and `eval.jsonl`.

#### Format of `train.jsonl` and `eval.jsonl`

Each line in the `jsonl` files will represent a single QA pair. The structure looks like this:

```json
{
  "question": "What is the cohesive energy of Sn?",
  "answer": "The cohesive energy of Sn is 0.04 eV/atom.",
  "context_ids": ["1601.05528v1_p1_c3", "1601.05528v1_p2_c1"],
  "doc_refs": ["1601.05528v1.pdf"]
}
```

* `question`: The question related to the battery chemistry topic.
* `answer`: The correct answer to the question.
* `context_ids`: A list of chunk IDs that contain relevant context from the paper.
* `doc_refs`: References to the documents (papers) from which the context is retrieved.

You can use `generate_qa.py` to generate these pairs from your processed papers.

### 2.2 Split Dataset

Make sure your dataset is split into two parts:

* `train.jsonl`: Used for training the model.
* `eval.jsonl`: Used for evaluation and validation.

You can use an 80/20 split between training and evaluation datasets.

---

## 3. Training the Retriever Model

### 3.1 Retriever Model

The first model to train is the **retriever**, which selects relevant chunks based on the input question. Here, we use **BatteryBERT** fine-tuned with a **DPR-style dual encoder**.

To train the retriever, use the following script:

```bash
python scripts/train_retriever.py --chunks_dir data/processed/chunks --train data/qa/train.jsonl --base_model batterydata/batterybert-cased --out_dir models/retriever_bbert_dpr
pip install numpy==1.26.4
```

This script performs the following:

* Loads the paper chunks from `data/processed/chunks`.
* Loads the question-context pairs from `data/qa/train.jsonl`.
* Fine-tunes a dual-encoder model on these pairs, saving the model to `models/retriever_bbert_dpr`.

---

## 4. Training the Generator Model

### 4.1 Generator Model (Supervised Fine-Tuning)

Once the retriever is trained, you can train the **generator** model, which is responsible for generating answers using the retrieved context.

We use **QLoRA** for efficient supervised fine-tuning of the language model.

To train the generator, use the following script:

```bash
python scripts/train_generator_sft.py \
  --train data/qa/train.jsonl \
  --chunks_dir data/processed/chunks \
  --base_model mistralai/Mistral-7B-Instruct-v0.2 \
  --out_dir models/generator_lora

# for testing the model and dataset
python scripts/train_retriever.py --chunks_dir data/processed/chunks --train data/qa/train_fixed.jsonl --base_model sentence-transformers/all-MiniLM-L6-v2 --out_dir models/test_retriever_fixed

# if chunks are not working
python scripts/fix_chunk_ids.py
```

This script does the following:

* Loads the question-answer pairs from `data/qa/train.jsonl`.
* Fine-tunes the **Mistral-7B-Instruct** model with LoRA for efficiency.
* Saves the fine-tuned model to `models/generator_lora`.

---

## 5. Evaluation

### 5.1 RAG Evaluation

After training, it’s essential to evaluate your model’s performance. You can use **RAGAS** for evaluating the retrieval accuracy, faithfulness, and groundedness of the generated answers.

Run the evaluation script in Jupyter:

```bash
python scripts/eval_ragas.py \
    --eval_file data/qa/eval.jsonl \
    --model_dir models/generator_lora \
    --chunk_dir data/processed/chunks
```

This notebook performs the following:

1. Loads the evaluation dataset (`eval.jsonl`).
2. Retrieves the top-k context chunks for each question using the fine-tuned retriever.
3. Generates answers based on the retrieved contexts.
4. Evaluates the generated answers for **accuracy**, **faithfulness**, **precision**, **recall**, and **relevance**.

---

## 6. Key Metrics to Monitor

* **Recall\@k**: Measures the ability of the retriever to retrieve relevant chunks. The goal is to achieve at least 0.85 for *k=5*.
* **Faithfulness**: Ensures that the model's generated answer is grounded in the retrieved context.
* **F1 Score / Exact Match**: Used for evaluating numeric or short answers.

---

## 7. Fine-Tuning and Hyperparameter Adjustment

* **Learning Rate**: Tune the learning rate in the retriever and generator models. Start with a smaller rate like `2e-4` and adjust based on validation results.
* **Batch Size**: Adjust the batch size depending on your GPU memory. For the A40, you can try a batch size of 2-4 for the generator.
* **Epochs**: Run 3-5 epochs for the retriever model and 2-3 epochs for the generator model.

---

## 8. Next Steps

* Improve model accuracy by adding more domain-specific data (e.g., new battery-related papers).
* Experiment with different retrieval architectures (e.g., cross-encoder reranking).
* Integrate more specialized models for other sub-domains within battery chemistry.

---

For further inquiries or troubleshooting, feel free to reach out!
