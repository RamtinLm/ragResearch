import gradio as gr
import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

load_dotenv()

def load_query_engine():
    collection_name = os.getenv("COLLECTION_NAME", "policy_docs")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    llm_model_name = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:3b")
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = OllamaEmbedding(model_name=embed_model_name, base_url=ollama_url)
    llm = Ollama(model=llm_model_name, base_url=ollama_url, request_timeout=120.0, context_window=2048)

    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    return index.as_query_engine(llm=llm, similarity_top_k=5)


query_engine = load_query_engine()

def ask(question):
    if not question.strip():
        return "Please enter a question.", ""
    response = query_engine.query(question)
    sources = "\n".join(
        f"• Score: {node.score:.3f} | {node.metadata.get('file_name', 'unknown')}"
        for node in response.source_nodes
    )
    return str(response), sources


with gr.Blocks(title="Policy Compliance Assistant") as demo:
    gr.Markdown("# 🍁 Policy Compliance Assistant")
    gr.Markdown("Ask questions about the **Safe Food for Canadians Regulations**. All processing is local — no data leaves this machine.")

    with gr.Row():
        question = gr.Textbox(label="Your Question", placeholder="e.g. What activities require a licence?", scale=4)
        submit = gr.Button("Ask", variant="primary", scale=1)

    answer = gr.Textbox(label="Answer", lines=5)
    sources = gr.Textbox(label="Sources", lines=4)

    submit.click(fn=ask, inputs=question, outputs=[answer, sources])
    question.submit(fn=ask, inputs=question, outputs=[answer, sources])


if __name__ == "__main__":
    demo.launch()