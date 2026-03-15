# Local RAG — Policy Compliance Assistant

A fully local AI assistant that checks food company SOPs against Government of Canada food regulations. No internet, no cloud, no data leaving the machine.

## What It Does

- **Phase 1 (current):** Answers natural language questions about indexed food policy documents
- **Phase 2 (in progress):** Accepts a user-uploaded SOP PDF and generates a structured compliance gap report, citing specific pages from both the SOP and the regulation where conflicts exist

## Tech Stack

| Layer | Tool |
|---|---|
| Embeddings | nomic-embed-text (via Ollama) |
| Vector Store | ChromaDB |
| LLM | Llama 3.2 3B (via Ollama) |
| Orchestration | LlamaIndex |
| Frontend | Gradio |
| Containers | Docker |

## Quick Start

### 1. Start containers
```bash
docker compose up -d
```

### 2. Pull models
```bash
docker exec -it ollama ollama pull llama3.2:3b
docker exec -it ollama ollama pull nomic-embed-text
```

### 3. Set up Python environment
```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Configure environment
```bash
cp .env.example .env
```

### 5. Ingest policy documents
Place PDFs in `data/raw/`, then:
```bash
python3 src/ingest.py
```

### 6. Launch the UI
```bash
python3 src/app.py
```

Open `http://127.0.0.1:7860` in your browser.

## Project Status

- [x] Docker environment (Ollama + ChromaDB)
- [x] Ingestion pipeline — PDF → chunks → vectors → ChromaDB
- [x] Query pipeline — question → retrieval → LLM answer
- [x] Gradio demo UI
- [ ] SOP upload + compliance gap analysis (in progress)
- [ ] RAGAS evaluation

## Notes

- Python 3.11 required (3.14 incompatible with ChromaDB)
- Docker memory limit: 10GB recommended on 16GB machines
- Re-ingestion wipes the collection first to avoid duplicates
