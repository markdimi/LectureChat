<h1 align="center">LectureChat</h1>
<p align="center">Grounded answers from Wikipedia and your lecture collections.</p>


# Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
  - [System Requirements](#system-requirements)
  - [Install Dependencies](#install-dependencies)
  - [Configure the LLM of Your Choice](#configure-the-llm-of-your-choice)
  - [Configure Information Retrieval](#configure-information-retrieval)
    - [Option 1 (Default): Use our free rate-limited Wikipedia search API](#option-1-default-use-our-free-rate-limited-wikipedia-search-api)
    - [Option 2: Add your lecture indices with FAISS (recommended here)](#option-2-add-your-lecture-indices-with-faiss-recommended-here)
      - [1) Prepare your data (JSONL)](#1-prepare-your-data-jsonl)
      - [2) Convert to Parquet](#2-convert-to-parquet)
      - [3) Build a FAISS index](#3-build-a-faiss-index)
      - [4) Register indices](#4-register-indices)
      - [5) Verify retrieval](#5-verify-retrieval)
- [Run LectureChat in Terminal](#run-lecturechat-in-terminal)
- [[Optional] Deploy LectureChat for Multi-user Access](#optional-deploy-lecturechat-for-multi-user-access)
  - [Set up Cosmos DB](#set-up-cosmos-db)
  - [Run Chainlit](#run-chainlit)
- [The Free Rate-limited Wikipedia Search API](#the-free-rate-limited-wikipedia-search-api)
- [WikiChat indexing and Wikipedia preprocessing](#wikichat-indexing-and-wikipedia-preprocessing)
- [Other Commands](#other-commands)
  - [Run a Distilled Model for Lower Latency and Cost](#run-a-distilled-model-for-lower-latency-and-cost)
  - [Simulate Conversations](#simulate-conversations)
- [License](#license)


# Introduction

LectureChat is a retrieval-augmented chatbot that combines Wikipedia search with FAISS-based search over your lecture collections (e.g., video transcripts). It is based on and extends WikiChat: https://github.com/stanford-oval/WikiChat

What is WikiChat? A high-quality RAG pipeline that grounds answers on Wikipedia using a multi-stage process: generate queries/claims → retrieve → filter → draft → refine. We inherit this pipeline and add a dedicated lecture retrieval branch.

How this system works (high level):
- **Wikipedia branch**: Uses the public multilingual Wikipedia search API (or a custom endpoint) and shows numeric citations [1], [2], …
- **Lecture branch (FAISS)**: Searches your configured FAISS indices, deduplicates, reranks, groups by time overlap (IoU), selects best, filters, and drafts a unified answer with letter citations [a], [b], … Separate lecture references include both a “Summary” and the “Original transcript”.
- The frontend displays the Wikipedia answer first, then a separate “Lectures” answer with letter-cited references.

If you prefer the original WikiChat approach to indexing (Qdrant, TEI, etc.), please refer to the WikiChat README. This README focuses on the FAISS-based approach for lectures.


# Installation

Installing LectureChat involves the following steps:

1. Install dependencies
1. Configure the LLM of your choice. LectureChat supports many LLMs (OpenAI, Azure, Anthropic, Mistral, HuggingFace, Together.ai, Groq).
1. Choose information retrieval sources:
   - Use our free, rate-limited Wikipedia API (default)
   - Add FAISS indices for your lectures (recommended here)
1. Run LectureChat with your desired configuration.
1. [Optional] Deploy LectureChat for multi-user access.


## System Requirements
Tested with Python 3.11 on Ubuntu 20.04 LTS. Other Linux distros should work; macOS/WSL may require extra troubleshooting.

Hardware:
- **Basic usage**: Minimal. Uses LLM APIs and the public Wikipedia API by default.
- **Local FAISS indices**: Ensure enough disk space and use SSD/NVMe for low latency.
- **Local LLM**: GPU required if serving models locally.
- **Creating new indices**: A GPU is recommended for fast embedding. The default embedding we use in code examples (Snowflake Arctic) can also run on CPU, but GPU is faster.


## Install Dependencies

Clone this repository and enter the project directory:
```bash
git clone https://github.com/markdimi/LectureChat.git
cd LectureChat
```

We recommend using the pixi environment defined in `pixi.toml`:
```bash
pixi shell
# Optional for user simulation only
python -m spacy download en_core_web_sm
```

This repository uses [`invoke`](https://www.pyinvoke.org/) for common tasks. List available tasks with:
```bash
inv -l
```
See details for a task with:
```bash
inv <task> --help
```

Docker is only required if you follow the WikiChat indexing path (Qdrant/TEI). It is not required for FAISS indexing used here.


## Configure the LLM of Your Choice

Edit `llm_config.yaml` and set up your preferred engine. Then create an `API_KEYS` file (gitignored) with your provider keys, e.g.:
```bash
OPENAI_API_KEY=...
MISTRAL_API_KEY=...
```

Locally hosted models do not require an API key but need an OpenAI-compatible endpoint.


## Configure Information Retrieval

### Option 1 (Default): Use our free rate-limited Wikipedia search API
By default the code uses https://search.genie.stanford.edu/wikipedia_20250320/ and requires no setup.

### Option 2: Add your lecture indices with FAISS (recommended here)

LectureChat reads FAISS indices via `retrieval/faiss_indices.json` and expects the index files and their Parquet metadata in `segment_indeces/`. Multiple indices are encouraged.

Recommended folder structure:
```
segments/                # your chunked data (JSONL/Parquet)
segment_indeces/         # FAISS index files (.faiss) and matching metadata (.parquet)
retrieval/faiss_indices.json
```

#### 1) Prepare your data (JSONL)
Each line should follow this schema (fields used by the UI and pipeline):
```json
{
  "id": 123,
  "document_title": "Intro to ML",
  "section_title": "Logistic Regression",
  "content": "Chunked transcript text...",
  "language": "en",
  "block_metadata": {
    "course_name": "ML 101",
    "course_term": "WS2024",
    "video_id": 45,
    "segment_index": 17,
    "start_ms": 123450,
    "end_ms": 129000,
    "start_sec": 123.45,
    "end_sec": 129.0
  }
}
```
Notes:
- `content` is the chunk text. Chunk to ~500 tokens for best retrieval.
- `language` is used for display flags in the UI (e.g., en/de).
- `block_metadata` fields power the lecture references (course info, video/segment/time).

If your data is not in JSONL yet, transform it with your own script into this format.

#### 2) Convert to Parquet
Use the included helper to bulk-convert JSONL files to Parquet in-place:
```bash
python preprocessing/convert_jsonl_to_parquet.py --input_dir ./segments
```
This will produce `*.parquet` files next to your JSONL files.

#### 3) Build a FAISS index
Create a FAISS index that matches the model used for querying in `retrieval/faiss_segments.py` (Snowflake Arctic, matryoshka 256-dim cosine/inner-product). The snippet below builds an index from a Parquet file and writes both the index and a filtered metadata parquet (same rows order) to `segment_indeces/`.

Install prerequisites (if not already present in your environment):
```bash
pip install faiss-cpu transformers torch pyarrow pandas tqdm
```

Create a script (e.g., `scripts/build_faiss_index.py`) with:
```python
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import faiss  # type: ignore
except Exception:
    import faiss_cpu as faiss  # type: ignore

from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "Snowflake/snowflake-arctic-embed-l-v2.0"

def embed_texts(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    out_embeddings = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            out = model(**enc)
            emb = out[0][:, 0]  # CLS pooling per model docs
            emb = emb.detach().cpu().numpy().astype(np.float32)
            # Matryoshka 256 dims + L2 normalize to match retrieval
            emb = emb[:, :256]
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
            out_embeddings.append(emb)
    return np.vstack(out_embeddings)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_parquet", required=True)
    ap.add_argument("--index_name", required=True, help="e.g., audio | gpt | texttiling | custom")
    ap.add_argument("--out_dir", default="segment_indeces")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load rows
    table = pq.read_table(args.input_parquet)
    df = table.to_pandas()

    # Build embedding input (title + section + content)
    def full_text(row):
        dt = str(row.get("document_title", "") or "")
        st = str(row.get("section_title", "") or "")
        c = str(row.get("content", "") or "")
        title = f"{dt} > {st}" if dt and st else dt or st
        return f"Title: {title}\n{c}".strip()

    texts = [full_text(r) for r in df.to_dict("records")]
    embs = embed_texts(texts)

    # Build FAISS index (cosine via inner product on normalized vectors)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs)

    # Persist
    faiss_path = out_dir / f"{args.index_name}.faiss"
    meta_path = out_dir / f"{args.index_name}.parquet"
    faiss.write_index(index, str(faiss_path))

    # Keep only required columns (others are allowed, but rows order must match index)
    keep_cols = [
        "id", "document_title", "section_title", "content", "language", "block_metadata",
        # optional convenience fields if present
        "course_name", "course_term", "video_id", "segment_index", "start_ms", "end_ms", "start_sec", "end_sec",
    ]
    cols = [c for c in df.columns if c in keep_cols]
    pq.write_table(pa.Table.from_pandas(df[cols], preserve_index=False), str(meta_path))

    print(f"Wrote: {faiss_path}")
    print(f"Wrote: {meta_path}")

if __name__ == "__main__":
    main()
```

Example usage:
```bash
python scripts/build_faiss_index.py \
  --input_parquet segments/chunked_gpt.parquet \
  --index_name gpt \
  --out_dir segment_indeces
```

Repeat for as many indices as you like.

#### 4) Register indices
Edit `retrieval/faiss_indices.json` and register your indices. Example:
```json
[
  {
    "name": "audio",
    "index_path": "segment_indeces/audio.faiss",
    "metadata_path": "segment_indeces/audio.parquet",
    "description": "Embedding index over audio-transcribed segments."
  },
  {
    "name": "gpt",
    "index_path": "segment_indeces/gpt.faiss",
    "metadata_path": "segment_indeces/gpt.parquet",
    "description": "Embedding index over GPT-generated segments."
  }
]
```
Notes:
- Paths may be relative; the code resolves them relative to the repository root.
- Multiple indices are encouraged. The system queries all and aggregates results.

#### 5) Verify retrieval
You can sanity-check FAISS retrieval from the Python REPL:
```bash
python - <<'PY'
from retrieval.faiss_segments import retrieve_all_indices
hits = retrieve_all_indices("What is backpropagation?", top_k=3)
print(hits[:3])
PY
```


# Run LectureChat in Terminal

Run different configurations with `invoke`:
```bash
inv demo --engine gpt-4o
```
The UI will show the Wikipedia answer first (numeric citations) and then the Lecture answer (letter citations) when FAISS indices are configured.


## Run Chainlit
Starts both backend and frontend; default port 5001:
```bash
inv chainlit --backend-port 5001
```
Use the above if you want to deploy the whole system.

# The Free Rate-limited Wikipedia Search API
For prototyping, you can use the public API documented at https://search.genie.stanford.edu/redoc. No guarantees; not production-grade.


# WikiChat indexing and Wikipedia preprocessing
If you want to follow the original WikiChat approach (Qdrant/TEI, Wikipedia preprocessing, uploading to 🤗 Hub, etc.), and also want to deploy the system for multi-user access, please refer to the WikiChat README and documentation: https://github.com/stanford-oval/WikiChat


# Other Commands

## Run a Distilled Model for Lower Latency and Cost
WikiChat>=2.0 is not compatible with the older fine-tuned LLaMA-2 checkpoints; refer to WikiChat v1.0 for distilled models.

## Simulate Conversations
You can simulate dialogues for evaluation:
```bash
inv simulate-users --num-dialogues 1 --num-turns 2 --simulation-mode passage --language en --subset head
```
Results will be saved in `benchmark/simulated_dialogues/`.


# License
LectureChat code, models, and data are released under the Apache-2.0 license (same as WikiChat).
