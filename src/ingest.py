import os
import chromadb
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

load_dotenv()

def ingest_documents():
    data_path = os.getenv("DATA_PATH", "data/raw")
    collection_name = os.getenv("COLLECTION_NAME", "policy_docs")
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embed_model_name = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    chroma_host = os.getenv("CHROMA_HOST", "localhost")
    chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

    print("🔄 Starting ingestion pipeline...")
    print(f"📄 Loading documents from {data_path}...")
    documents = SimpleDirectoryReader(data_path).load_data()
    print(f"✅ Loaded {len(documents)} document(s)")

    embed_model = OllamaEmbedding(
        model_name=embed_model_name,
        base_url=ollama_url
    )

    print("🗄️ Connecting to ChromaDB...")
    chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("⚙️ Chunking and embedding documents...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )

    print("✅ Ingestion complete! Documents stored in ChromaDB.")
    return index

if __name__ == "__main__":
    ingest_documents()