from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.obsidian import ObsidianReader
from llama_index.vector_stores import ChromaVectorStore
from llama_index.graph_stores import SimpleGraphStore
from llama_index import QueryEngine
import chromadb

# Load your Obsidian Markdown notes
reader = reader = ObsidianReader(
    input_dir="/Users/chris/zk-copy-for-testing/0-Current Focus"
)
documents = reader.load_data()

# # Set up ChromaDB as the vector store
# chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage
# vector_store = ChromaVectorStore(chroma_client, collection_name="obsidian_notes")

# # Create LlamaIndex with ChromaDB as backend
# index = VectorStoreIndex.from_documents(documents, vector_store=vector_store)

# Initialize a simple graph store
# graph_store = SimpleGraphStore()

# Create a Knowledge Graph Index
# graph_index = KnowledgeGraphIndex.from_documents(documents, graph_store=graph_store)


# Create query engines for both vector and graph retrieval
# vector_query_engine = index.as_query_engine()
# graph_query_engine = graph_index.as_query_engine()

# # Hybrid query engine
# query_engine = QueryEngine(vector_query_engine, graph_query_engine)

# response = query_engine.query("How do I optimize dataset shuffling in deep learning?")
# print(response)
