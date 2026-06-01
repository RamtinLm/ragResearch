#!/usr/bin/env python3
"""
Fact-check script for Policy Compliance Assistant RAG system.
Run from REPO ROOT: python verify.py

Sections 1-7 check source code and files — no Docker needed.
Section 8 checks live infrastructure — Docker must be running.
"""

import json
import urllib.request
from pathlib import Path

# ── Helpers ──────────────────────────────────────────────────────────

results = []

def check(label, condition, detail=""):
    icon = "✅" if condition else "❌"
    tag  = "PASS" if condition else "FAIL"
    results.append((tag, label, detail))
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon} {label}{suffix}")

def section(title):
    print(f"\n{'─'*55}")
    print(f"  {title}")
    print(f"{'─'*55}")

def src(filename):
    try:
        return Path(f"src/{filename}").read_text()
    except FileNotFoundError:
        return ""

def http_get(url, timeout=4):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    except Exception:
        return None

def http_post(url, payload, timeout=12):
    try:
        data = json.dumps(payload).encode()
        req  = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
#  SECTION 1 — File & directory structure
# ══════════════════════════════════════════════════════════════════════
section("1. File & Directory Structure")

check("src/ exists",                    Path("src").is_dir())
check("src/__init__.py exists",         Path("src/__init__.py").is_file())
check("src/app.py exists",              Path("src/app.py").is_file())
check("src/compliance.py exists",       Path("src/compliance.py").is_file())
check("src/ingest.py exists",           Path("src/ingest.py").is_file())
check("src/query.py exists",            Path("src/query.py").is_file())
check("data/raw/ exists",               Path("data/raw").is_dir())
check("data/processed/ exists",         Path("data/processed").is_dir())
check("tests/ exists",                  Path("tests").is_dir())
check("docker-compose.yml exists",      Path("docker-compose.yml").is_file())
check(".env exists",                    Path(".env").is_file())
check(".env.example exists",            Path(".env.example").is_file())
check("requirements.txt exists",        Path("requirements.txt").is_file())

pdfs = list(Path("data/raw").glob("*.pdf"))
check("regulations PDF in data/raw/",
      len(pdfs) > 0,
      pdfs[0].name if pdfs else "no PDF found")

# Exclude verify.py itself from this check
root_pys = [f.name for f in Path(".").glob("*.py") if f.name != "verify.py"]
check("No .py source files at repo root (src/ only)",
      not root_pys,
      f"found: {root_pys}" if root_pys else "clean")


# ══════════════════════════════════════════════════════════════════════
#  SECTION 2 — docker-compose.yml
# ══════════════════════════════════════════════════════════════════════
section("2. docker-compose.yml")

dc = Path("docker-compose.yml").read_text() if Path("docker-compose.yml").exists() else ""

check("Ollama image: ollama/ollama",             "image: ollama/ollama" in dc)
check("Ollama port: 11434:11434",                "11434:11434" in dc)
check("Ollama volume: ollama_data:/root/.ollama","ollama_data:/root/.ollama" in dc)
check("ChromaDB image: chromadb/chroma",         "image: chromadb/chroma" in dc)
check("ChromaDB port: 8000:8000",                "8000:8000" in dc)
check("ChromaDB volume: chroma_data:/chroma",    "chroma_data:/chroma/chroma" in dc)
check("restart: unless-stopped on both",         dc.count("restart: unless-stopped") == 2)
check("Named volumes declared",                  "ollama_data:" in dc and "chroma_data:" in dc)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 3 — .env.example keys
# ══════════════════════════════════════════════════════════════════════
section("3. .env.example Keys")

env_ex = Path(".env.example").read_text() if Path(".env.example").exists() else ""

for key in ["OLLAMA_BASE_URL", "OLLAMA_EMBED_MODEL", "OLLAMA_LLM_MODEL",
            "CHROMA_HOST", "CHROMA_PORT", "COLLECTION_NAME"]:
    check(f"{key} present", key in env_ex)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 4 — ingest.py
# ══════════════════════════════════════════════════════════════════════
section("4. src/ingest.py")

ing = src("ingest.py")

check("Uses SimpleDirectoryReader",             "SimpleDirectoryReader" in ing)
check("chunk_size=128",                         "chunk_size=128" in ing)
check("chunk_overlap=20",                       "chunk_overlap=20" in ing)
check("Word filter direction: <= 300",          "<= 300" in ing,
      "removes chunks OVER 300 words")
check("embed_batch_size=1",                     "embed_batch_size=1" in ing)
check("DATA_PATH default: data/raw",            '"data/raw"' in ing)
check("COLLECTION_NAME default: policy_docs",   '"policy_docs"' in ing)
check("Embed model default: nomic-embed-text",  '"nomic-embed-text"' in ing)
check("StorageContext used for ChromaDB",        "StorageContext" in ing)
check("show_progress=True",                     "show_progress=True" in ing)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 5 — query.py
# ══════════════════════════════════════════════════════════════════════
section("5. src/query.py")

qry = src("query.py")

check("context_window=2048",                     "context_window=2048" in qry)
check("similarity_top_k=5",                      "similarity_top_k=5" in qry)
check("request_timeout=120.0",                   "request_timeout=120.0" in qry)
check("COLLECTION_NAME default: policy_docs",    '"policy_docs"' in qry)
check("LLM default: llama3.2:3b",               '"llama3.2:3b"' in qry,
      "stale llama3.1:8b was fixed in pre-phase3 cleanup")
check("from_vector_store() used (reads, not re-ingests)",
      "from_vector_store" in qry)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 6 — app.py
# ══════════════════════════════════════════════════════════════════════
section("6. src/app.py")

app = src("app.py")

check("Imports check_compliance from compliance",
      "from compliance import check_compliance" in app)
check("LLM default: llama3.2:3b",               '"llama3.2:3b"' in app)
check("context_window=2048",                     "context_window=2048" in app)
check("similarity_top_k=5",                      "similarity_top_k=5" in app)
check("request_timeout=120.0",                   "request_timeout=120.0" in app)
check("query_engine initialized at module level",
      "query_engine = load_query_engine()" in app)
check("Tab: Q&A",                                '"Q&A"' in app)
check("Tab: SOP Compliance Check",               '"SOP Compliance Check"' in app)
check("File upload accepts .pdf only",           '".pdf"' in app or "'.pdf'" in app)
check("pdf_file.name passed to check_compliance","pdf_file.name" in app)
check("gr.Progress() used",                      "gr.Progress()" in app)
check("from_vector_store() used",                "from_vector_store" in app)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 7 — compliance.py
# ══════════════════════════════════════════════════════════════════════
section("7. src/compliance.py")

com = src("compliance.py")

check("Uses pypdf PdfReader (not SimpleDirectoryReader)",
      "PdfReader" in com and "SimpleDirectoryReader" not in com)
check("chunk_size=128",                          "chunk_size=128" in com)
check("chunk_overlap=20",                        "chunk_overlap=20" in com)
check("Word filter direction: >= 20",            ">= 20" in com,
      "removes chunks UNDER 20 words (opposite of ingest.py)")
check("similarity_top_k=3 (not 5)",             "similarity_top_k=3" in com)
check("request_timeout=180.0 (not 120.0)",      "request_timeout=180.0" in com)
check("context_window=2048",                    "context_window=2048" in com)
check("LLM default: llama3.2:3b",              '"llama3.2:3b"' in com)
check("Uses llm.complete() for generation",     "llm.complete(prompt)" in com)
check("Does NOT use query_engine.query()",      "query_engine.query(" not in com)
check("NON-COMPLIANT keyword in prompt",        "NON-COMPLIANT" in com)
check("sop_page in metadata",                   '"sop_page"' in com)
check("SOP text preview: chunk.text[:200]",     "chunk.text[:200]" in com)
check("Skips chunks with no policy nodes",      "if not policy_nodes" in com)
check("format_report() function exists",        "def format_report" in com)


# ══════════════════════════════════════════════════════════════════════
#  SECTION 8 — Live infrastructure (Docker must be running)
#  ChromaDB uses v2 API — /api/v1/ endpoints return 404/deprecated error
# ══════════════════════════════════════════════════════════════════════
section("8. Live Infrastructure  (skip if Docker is off)")

V2 = "http://localhost:8000/api/v2"

# Ollama
ollama_raw = http_get("http://localhost:11434/")
check("Ollama responds at localhost:11434",
      ollama_raw is not None,
      ollama_raw.strip() if ollama_raw else "not reachable")

# ChromaDB — use v2 identity endpoint (v1 is deprecated in current chroma image)
chroma_raw = http_get(f"{V2}/auth/identity")
check("ChromaDB responds at localhost:8000 (v2 API)",
      chroma_raw is not None,
      "ok" if chroma_raw else "not reachable — or still on v1 image")

# Models
tags_raw   = http_get("http://localhost:11434/api/tags") if ollama_raw else None
tags_data  = json.loads(tags_raw) if tags_raw else {}
model_names = [m.get("name", "") for m in tags_data.get("models", [])]

check("llama3.2:3b pulled in Ollama",
      any("llama3.2:3b" in n for n in model_names),
      f"found: {model_names}" if model_names else "Ollama unreachable")

check("nomic-embed-text pulled in Ollama",
      any("nomic-embed-text" in n for n in model_names),
      f"found: {model_names}" if model_names else "Ollama unreachable")

# Embedding dimension
emb = http_post("http://localhost:11434/api/embeddings",
                {"model": "nomic-embed-text", "prompt": "test"}) if ollama_raw else None
if emb and "embedding" in emb:
    dim = len(emb["embedding"])
    check("nomic-embed-text → 768-dim vector", dim == 768, f"got {dim}")
else:
    check("nomic-embed-text → 768-dim vector", False,
          "Ollama unreachable or model not pulled")

# ChromaDB collection — v2 path: /tenants/default_tenant/databases/default_database/collections
col_url  = f"{V2}/tenants/default_tenant/databases/default_database/collections"
cols_raw = http_get(col_url) if chroma_raw else None

if cols_raw:
    try:
        cols       = json.loads(cols_raw)
        policy_col = next((c for c in cols if c.get("name") == "policy_docs"), None)
        check("policy_docs collection exists in ChromaDB", policy_col is not None,
              f"collections found: {[c.get('name') for c in cols]}")

        if policy_col:
            col_id    = policy_col.get("id")
            count_raw = http_get(f"{col_url}/{col_id}/count")
            if count_raw:
                count = json.loads(count_raw)
                # v2 returns a plain integer, not a dict
                count_val = count if isinstance(count, int) else count.get("count", 0)
                check("policy_docs has vectors (count > 0)",
                      count_val > 0, f"count = {count_val}")
            else:
                check("policy_docs has vectors (count > 0)", False, "count endpoint failed")
        else:
            check("policy_docs has vectors (count > 0)", False,
                  "collection not found — run: python src/ingest.py")
    except Exception as e:
        check("policy_docs collection exists in ChromaDB", False, str(e))
        check("policy_docs has vectors (count > 0)", False)
else:
    check("policy_docs collection exists in ChromaDB", False,
          "ChromaDB unreachable — run: docker compose up -d")
    check("policy_docs has vectors (count > 0)", False)


# ══════════════════════════════════════════════════════════════════════
#  SUMMARY
# ══════════════════════════════════════════════════════════════════════
passed = [r for r in results if r[0] == "PASS"]
failed = [r for r in results if r[0] == "FAIL"]

print(f"\n{'═'*55}")
print(f"  RESULT: {len(passed)}/{len(results)} checks passed")
print(f"{'═'*55}")

if failed:
    print("\n  ❌ Failed checks:")
    for _, label, detail in failed:
        suffix = f"  ({detail})" if detail else ""
        print(f"     • {label}{suffix}")
else:
    print("\n  ✅ All checks passed!")

print()
