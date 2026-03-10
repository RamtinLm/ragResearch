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
    llm_model_name = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b")
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        base_url=ollama_url
    )

    llm = Ollama(
        model=llm_model_name,
        base_url=ollama_url,
        request_timeout=120.0,
        context_window=2048
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
    return query_engine


def ask(question: str):
    print(f"\n🔍 Question: {question}")
    query_engine = load_query_engine()
    response = query_engine.query(question)
    print(f"\n📋 Answer:\n{response}")
    print(f"\n📎 Sources:")
    for node in response.source_nodes:
        print(f"  - Score: {node.score:.3f} | {node.metadata.get('file_name', 'unknown')}")
    return response


if __name__ == "__main__":
    question = input("Ask a question about the policy: ")
    ask(question)