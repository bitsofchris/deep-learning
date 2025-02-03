from pathlib import Path

from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.storage import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

# Build an index on the combined documents.
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5", device="cpu"  # Use "cuda" for GPU acceleration
)

# Define the directory where the index is stored
persist_dir = "./storage"  # Change this to your actual storage directory

# Load the index from storage
storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
index = load_index_from_storage(storage_context)

# # Access the nodes (embedded chunks) from the index
retriever = index.as_retriever()
nodes = retriever.store.docs  # This retrieves all stored nodes

print(f"Loaded {len(nodes)} nodes from the index.")
