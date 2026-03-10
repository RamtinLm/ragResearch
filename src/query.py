import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

def load_query_engine(collection_name: str = "policy_docs"):
    # 1. Connect to ChromaDB
    chroma_client = chromadb.HttpClient(host="localhost", port=8000)
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 2. Set up embedding model
    embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434"
    )

    # 3. Set up LLM
    llm = Ollama(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        request_timeout=120.0
    )

    # 4. Load index from existing ChromaDB
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    # 5. Build query engine
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