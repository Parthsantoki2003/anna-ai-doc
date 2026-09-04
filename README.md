# Anna Document Workspace: Production-Grade RAG Engine

A high-performance, hybrid Retrieval-Augmented Generation (RAG) system packaged for the Anna OS App model. It provides production-grade document processing, intelligent hybrid search, and source-grounded response generation.

## Anna OS packaging

This project follows Anna's current schema-2 App model:

* `manifest.json` declares the App UI, prompt addendum, permissions, and views.
* `app.json` contains Anna App Store listing metadata.
* `bundle/index.html` is the static SPA entry required by `anna-app validate`.
* The FastAPI service remains the local/cloud backend for document ingestion and retrieval.

Anna Apps use Anna-provided model access or BYOK configuration; this project does not depend on an unverified Python Anna SDK.

## 🚀 Key Features

* **Hybrid Retrieval:** Combines **BM25 (Sparse)** search for keyword matching with **Qdrant (Dense)** vector search for semantic understanding, fused via **Reciprocal Rank Fusion (RRF)**.
* **Efficient Hybrid Retrieval:** Uses BM25 and Qdrant semantic search with Reciprocal Rank Fusion, keeping the deployment lightweight.
* **Intelligent Extraction:** Handles multi-format PDFs with automatic OCR fallback via **Tesseract** and **PyMuPDF**.
* **Dynamic Chunking:** Configurable chunking strategies (Recursive vs. Fixed) with adjustable overlap via API parameters.
* **Production-Ready:** Built with FastAPI, containerized with Docker, and optimized with Gunicorn workers for deployment.
* **LLM-Powered Generation:** Leverages Llama 3.3 (via Groq) with source citations to prevent hallucinations.

## 🛠 Tech Stack

* **Framework:** FastAPI
* **Vector Database:** Qdrant
* **Search Algorithms:** BM25, RRF, Cross-Encoder
* **Models:** `all-MiniLM-L6-v2` (Embeddings), Llama-3.3-70b (LLM)
* **OCR:** Tesseract + PyMuPDF
* **Deployment:** Docker, Gunicorn, Render/GCP ready

## 📋 Architecture
The system follows a refined RAG pipeline:
1. **Ingest:** Extract → Chunk → Embed → Upsert to Qdrant.
2. **Retrieve:** Parallel Sparse (BM25) + Dense (Vector) search.
3. **Refine:** Reciprocal Rank Fusion (RRF).
4. **Generate:** Context-augmented response from LLM with source citations.

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone (https://github.com/Parthsantoki2003/Intelligent-Documen-QA-API.git)
cd "Intelligent-Documen-QA-API"
```
2. Install Dependencies
```bash

pip install -r requirements.txt
```
3. Environment Setup
Create a .env file in the root directory:

```bash
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
ANNA_APP_SLUG=anna-document-workspace
```
4. Run the API

```bash
python main.py
```

To validate the Anna package, install the current `@anna-ai/cli` and run:

```bash
anna-app validate --strict
```
