# PubHealth Medical Claim Verification

This repository contains a retrieval-augmented medical/public-health claim verification pipeline built around the PubHealth dataset. The system retrieves evidence for each claim, formats the retrieved evidence into a classifier input, and trains/evaluates encoder-based models such as SciBERT and Longformer.

The main task is four-class claim classification:

- `false`
- `mixture`
- `true`
- `unproven`

The project also includes utilities for three-class experiments, no-RAG baselines, validation threshold tuning, judge-LLM evidence diagnostics, and EDA figure generation.

## Project Overview

The pipeline is organized around three main stages:

1. **RAG dataset construction**
   - Loads PubHealth train/validation/test splits.
   - Builds a retrieval index over evidence text or an external health corpus.
   - Retrieves the top evidence chunks for each claim.
   - Saves each processed example as JSONL.

2. **Classifier training**
   - Uses HuggingFace Transformers.
   - Supports SciBERT-style 512-token inputs and longer-context models.
   - Supports RAG input (`claim + query + retrieved evidence`) and claim-only baselines.

3. **Inference and analysis**
   - Runs prediction on selected splits.
   - Saves prediction CSV files with probabilities and confidence values.
   - Supports threshold-grid experiments and visualization/EDA scripts.

The final reported model for the project was a four-class SciBERT RAG classifier using argmax inference.

## Repository Structure

```text
.
├── configs/
│
├── figures_scripts/
│   ├── pubhealth_eda.py                     # Main EDA/report figure generation
│   ├── model_comparison.py                  # Model comparison analysis
│   ├── scibert_token_window.py              # Token-window analysis
│   └── visualize_results.py                 # Additional result visualizations
│
├── inference/
│   ├── run_inference.py                     # Main inference script
│   └── judge_llm.py                         # Judge-LLM evidence diagnostic
│
├── query_reformulation/
│   ├── query_reformulation.py               # Query reformulation code
│   └── query_reform_prompt.py               # Reformulation prompt
│
├── rag_dataset/
│   ├── build_rag_dataset.py                 # Builds processed RAG JSONL dataset
│   ├── rag.py                               # BM25 + dense + cross-encoder RAG implementation
│   ├── reformat_rag_dataset.py              # Dataset reformatting utility
│   └── make_unknown_dataset.py              # Converts 4-class data to 3-class unknown setup
│
├── utils/
│   ├── classification_datasets.py           # PyTorch datasets for RAG and claim-only input
│   ├── input_formatting.py                  # Builds classifier input text
│   ├── pubhealth_loader.py                  # Loads PubHealth parquet/processed JSONL data
│   ├── rag_builders.py                      # RAG builder utilities
│   ├── external_corpus.py                   # PubMedQA/Wikipedia external-corpus loading
│   ├── label_utils.py                       # Label mapping utilities
│   └── config_utils.py                      # Config loading utilities
│
├── grid_search_thresholds.py                # Validation threshold tuning
├── train_classifier.py                      # Main training implementation
├── main.py                                  # Main training entry point
├── requirements.txt
└── README.md
```

## Expected Data Layout

The PubHealth parquet files should be placed in the following structure:

```text
data/
├── pubhealth_bigbio_pairs/
│   ├── train/0000.parquet
│   ├── validation/0000.parquet
│   └── test/0000.parquet
│
└── pubhealth_source/
    ├── train/0000.parquet
    ├── validation/0000.parquet
    └── test/0000.parquet
```

Processed RAG datasets are saved under:

```text
data/processed/
```

Example processed files used during the project include:

```text
data/processed/FULL_no_reform.jsonl
data/processed/FULL_no_reform_3class_unknown.jsonl
```

## Environment Setup

Create and activate a virtual environment:

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Windows Command Prompt

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### macOS/Linux

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If PyTorch installation fails, install a PyTorch build compatible with your system/CUDA version first, then rerun the remaining requirements installation.

## Configuration Files

Most experiments are controlled through JSON config files in `configs/`.

The main config is:

```text
configs/config1.json
```

Important fields include:

```json
{
  "data": {
    "local_dir": "data",
    "processed_dataset_path": "data/processed/FULL_no_reform.jsonl",
    "max_examples_per_split": 10000
  },
  "rag": {
    "knowledge_base": "external_health",
    "query_source": "original",
    "top_k": 5,
    "method": "cross_encoder",
    "chunk_size": 150,
    "chunk_overlap": 30
  },
  "model": {
    "model_name": "allenai/scibert_scivocab_uncased",
    "max_length": 512,
    "input_mode": "claim_query_rag",
    "label_map": {
      "false": 0,
      "mixture": 1,
      "true": 2,
      "unproven": 3
    }
  }
}
```

## Building the RAG Dataset

Run the RAG dataset builder from the project root:

```bash
python rag_dataset/build_rag_dataset.py
```

By default, this uses:

```text
configs/config1.json
```

The output path is controlled by:

```json
"data": {
  "processed_dataset_path": "data/processed/FULL_no_reform.jsonl"
}
```

### RAG Method

The RAG pipeline supports:

- BM25 lexical retrieval
- dense bi-encoder retrieval using `BAAI/bge-small-en-v1.5`
- cross-encoder reranking using `cross-encoder/ms-marco-MiniLM-L-6-v2`

The retriever returns the top-ranked chunks for each claim. The processed JSONL stores the retrieved evidence in a structured format rather than only as one large string.

A processed example contains fields such as:

```json
{
  "split": "train",
  "example_id": "0",
  "claim": "...",
  "reformulated_query": "...",
  "rag_query": "...",
  "rag_context": [
    {
      "doc_number": 1,
      "doc_header": "[DOC 1 | score=9.4772]",
      "score": 9.4772,
      "text": "...",
      "metadata": {
        "source": "train_0",
        "chunking_method": "sentence_window",
        "word_count": 139
      },
      "method": "cross_encoder",
      "retrieved_by": ["biencoder", "bm25"],
      "bm25_score": 122.1652,
      "dense_score": 0.9024,
      "cross_encoder_score": 9.4772
    }
  ],
  "rag_results": [
    {
      "doc_number": 1,
      "doc_header": "[DOC 1 | score=9.4772]",
      "score": 9.4772,
      "text": "..."
    }
  ],
  "label": "false",
  "top_rag_score": 9.4772
}
```

This structure makes it easier to later decide how many complete chunks should be included in the model input.

## Chunking Strategy

The project uses sentence-window chunking. Documents are split into sentences, then complete sentences are grouped together until the chunk word budget is reached.

The chunker uses:

- `chunk_size`: maximum approximate words per chunk
- `chunk_overlap`: approximate overlap between consecutive chunks
- sentence-level overlap rather than arbitrary mid-sentence overlap

A sentence is split only if a single sentence is longer than `chunk_size`. Otherwise, chunks preserve sentence boundaries.

## Building Classifier Inputs

Classifier inputs are built in `utils/input_formatting.py` and used by `utils/classification_datasets.py`.

For RAG mode, the input format is:

```text
Original claim:
<claim>

Search query:
<rag_query>

Retrieved evidence:
<ranked evidence chunks>
```

For no-RAG mode, the input contains only the claim.

The classifier tokenizer handles truncation with the configured `max_length`. For SciBERT, this is typically 512 tokens.

## Training

To train using the main config, run:

```bash
python main.py
```

`main.py` loads:

```text
configs/config1.json
```

It creates a timestamped model output directory and calls:

```python
train_classifier_from_config(cfg)
```

Model checkpoints and tokenizer files are saved under the configured model output directory, for example:

```text
models/scibert_4class_rag_<timestamp>_<uuid>/
```

The training script also saves:

```text
label_map.json
```

inside the model directory.

## Running Inference

Run inference from the project root.

### Four-Class RAG Inference

Example:

```bash
python inference/run_inference.py \
  --dataset data/processed/FULL_no_reform_TEST.jsonl \
  --model-dir models/scibert_20260605_152114_fa6345e0-e3af-4933-bc23-0dbce6a63b12 \
  --mode rag \
  --num-classes 4 \
  --output data/predictions/scibert_4class_rag_test.csv \
  --splits test \
  --batch-size 4 \
  --max-length 512
```

### Claim-Only / No-RAG Inference

Example:

```bash
python inference/run_inference.py \
  --dataset data/processed/FULL_no_reform.jsonl \
  --model-dir models/scibert_no_rag_santi \
  --mode no_rag \
  --num-classes 4 \
  --output data/predictions/scibert_no_rag_test.csv \
  --splits test \
  --batch-size 4 \
  --max-length 512
```

The inference script prints:

- total examples
- accuracy
- macro-F1
- weighted-F1
- classification report
- confusion matrix

It also saves a prediction CSV containing:

- gold labels
- predicted labels
- model confidence
- class probabilities such as `prob_false`, `prob_mixture`, `prob_true`, and `prob_unproven`

## Three-Class Unknown Experiment

The three-class experiment combines:

```text
mixture + unproven -> unknown
```

Generate the three-class JSONL file with:

```bash
python rag_dataset/make_unknown_dataset.py
```

This reads:

```text
data/processed/FULL_no_reform.jsonl
```

and writes:

```text
data/processed/FULL_no_reform_3class_unknown.jsonl
```

Use a compatible three-class config/model when training or evaluating this setup.

## Threshold Grid Search

After inference probabilities are available, threshold tuning can be run with:

```bash
python grid_search_thresholds.py
```

This script runs inference once, then evaluates threshold rules over validation predictions. It saves outputs under:

```text
data/threshold_grid/
```

The most important output is:

```text
data/threshold_grid/threshold_grid_summary.csv
```

Thresholding was used as a diagnostic experiment. The final reported four-class model used argmax inference because it performed better on the processed test split.

## Judge LLM Evidence Diagnostic

The judge LLM diagnostic is implemented in:

```text
inference/judge_llm.py
```

The judge does not replace the classifier. It evaluates whether the retrieved evidence appears to support the classifier's predicted label.

The diagnostic produces fields such as:

```text
judge_verdict
judge_explanation
```

where the verdict is typically one of:

```text
supported
not_supported
```

This helps separate topical relevance from actual evidence support.

## EDA and Figure Generation

The `figures_scripts/` folder contains scripts used for report analysis.

### Main EDA

```bash
python figures_scripts/pubhealth_eda.py
```

Expected output:

```text
EDA/
```

### Model Comparison

```bash
python figures_scripts/model_comparison.py
```

Expected output:

```text
EDA/model_comparison/
```

### SciBERT Token-Window Analysis

```bash
python figures_scripts/scibert_token_window.py
```

This script helps analyze how claim length, retrieved-context length, and model context-window constraints interact.

## Main Project Takeaway

The project showed that RAG can improve PubHealth claim verification, but only when retrieval quality is carefully controlled. Sentence-window chunking, structured RAG outputs, saved processed datasets, and evidence-support diagnostics were important for making the model behavior interpretable and repeatable.
