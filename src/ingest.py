import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding

def ingest_documents(data_path: str = "data/raw", collection_name: str = "policy_docs"):
    print("🔄 Starting ingestion pipeline...")

    # 1. Load documents
    print(f"📄 Loading documents from {data_path}...")
    documents = SimpleDirectoryReader(data_path).load_data()
    print(f"✅ Loaded {len(documents)} document(s)")

    # 2. Set up embedding model (nomic via Ollama)
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    # 3. Set up ChromaDB
    print("🗄️ Connecting to ChromaDB...")
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4. Build index (chunks + embeds + stores)
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