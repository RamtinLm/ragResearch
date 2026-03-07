# Local RAG — Policy Compliance Assistant

A fully local AI assistant for answering questions about policy documents (Government of Canada food regulations). No internet, no cloud.

## Tech Stack
- **Docling** — PDF parsing
- **LlamaIndex** — Chunking & orchestration
- **nomic-embed-text** — Embeddings (via Ollama)
- **ChromaDB** — Vector store
- **Llama 3.1 8B** — LLM inference (via Ollama)
- **RAGAS** — Evaluation
- **Docker** — Containerization

## Quick Start

### 1. Start containers
\`\`\`bash
docker compose up -d
\`\`\`

### 2. Pull models
\`\`\`bash
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama pull nomic-embed-text
\`\`\`

### 3. Set up Python environment
\`\`\`bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Status
- [x] Tech stack finalized
- [x] Project structure initialized
- [ ] Ingestion pipeline
- [ ] Query pipeline
- [ ] Gradio demo UI
- [ ] Compliance validation test
