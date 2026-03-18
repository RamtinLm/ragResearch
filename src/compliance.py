import os
import chromadb
from pypdf import PdfReader
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

def load_components():
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    llm_model_name = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
    collection_name = os.getenv("COLLECTION_NAME", "policy_docs")

    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=ollama_url)
    llm = Ollama(model=llm_model_name, base_url=ollama_url, request_timeout=180.0, context_window=2048)

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=3)

    return retriever, llm


def extract_sop_chunks(pdf_path):
    """Read SOP PDF and split into chunks with page tracking."""
    reader = PdfReader(pdf_path)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(Document(text=text, metadata={"sop_page": i + 1}))

    splitter = SentenceSplitter(chunk_size=128, chunk_overlap=20)
    nodes = splitter.get_nodes_from_documents(documents)
    nodes = [n for n in nodes if len(n.text.split()) >= 20]  # skip tiny fragments
    return nodes


def check_compliance(pdf_path, progress=None):
    """Main compliance check — returns a full report as a string."""
    retriever, llm = load_components()
    sop_chunks = extract_sop_chunks(pdf_path)

    issues = []
    total = len(sop_chunks)

    for i, chunk in enumerate(sop_chunks):
        if progress:
            progress(i / total, desc=f"Checking chunk {i+1} of {total}...")

        # Retrieve relevant policy sections for this SOP chunk
        policy_nodes = retriever.retrieve(chunk.text)
        if not policy_nodes:
            continue

        policy_context = "\n\n".join(
            f"[Policy chunk {j+1}, score {n.score:.2f}]:\n{n.text}"
            for j, n in enumerate(policy_nodes)
        )

        sop_page = chunk.metadata.get("sop_page", "?")

        prompt = f"""You are a strict food safety compliance auditor checking a company SOP against Canadian food safety regulations.

SOP SECTION (page {sop_page}):
{chunk.text}

RELEVANT CANADIAN POLICY SECTIONS:
{policy_context}

Your job: Does this SOP section conflict with or fall short of the Canadian policy?

Rules:
- If the SOP describes a procedure that CONTRADICTS or is WEAKER than what the policy requires, reply with "NON-COMPLIANT: " followed by one sentence naming exactly what the SOP says vs what the policy requires.
- If the SOP section is about a topic the policy also covers and they AGREE, reply only with "COMPLIANT"
- ONLY reply "NOT APPLICABLE" if the SOP section is about something completely unrelated to food safety (e.g. company address, copyright notice, table of contents)
- Be strict. A weaker standard (e.g. longer time limits, optional steps, missing requirements) counts as NON-COMPLIANT.
- Do not give benefit of the doubt. If the SOP is vague where the policy is specific, that is NON-COMPLIANT.
"""

        response = llm.complete(prompt)
        result = str(response).strip()
        
        if result.startswith("NON-COMPLIANT"):
            issues.append({
                "sop_page": sop_page,
                "sop_text": chunk.text[:200],
                "issue": result
            })

    return format_report(issues)


def format_report(issues):
    if not issues:
        return "✅ No compliance issues found. The SOP appears to align with the Safe Food for Canadians Regulations."

    report = f"⚠️ Found {len(issues)} potential compliance issue(s):\n\n"
    report += "─" * 60 + "\n\n"

    for i, issue in enumerate(issues, 1):
        report += f"Issue #{i} — SOP Page {issue['sop_page']}\n"
        report += f"SOP Text: ...{issue['sop_text']}...\n"
        report += f"Finding: {issue['issue']}\n"
        report += "\n" + "─" * 60 + "\n\n"

    return report