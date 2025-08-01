# LangChain Chat with Your Data

https://learn.deeplearning.ai/courses/langchain-chat-with-your-data/lesson/1/introduction

# Intro
Components of lang chain
- prompts
- models
- indexes
- chains
- agents

### LangChain - Retrieval Augmented Generation

Document loading -> Splitting -> Storage -> retrieval -> output


##### Vector Store Loading
Document loading -> splitting - > storage (Vector store)
##### Retrieval
Query Storage (vector store) -> relevant splits -> Output  (prompt + LLM)


# Document Loading
- Variety of data -> standard document object
- langchain.document_loads import PyPDFLoader
YouTube
- OpenAiWhisperParser (audio to text)
- YouTubeAudio loader
URLs
- WebBaseLoader -> might need some post processing/ DE to clean it up
Notion Database
- Noteion_DB


# Document Splitting
- split chunks per lengths of each document
- text splitters - what text/ length
	- markdown header text splitter

Splitting chunks - some chunk size with some chunk overlap

MarkdownHeaderTextSplitter()
NLTKTextSplitter()

Chunk size measure in different ways:

### Vectorstores and Embedding

Create an index and store your chunks for look up

Embeddings  create numerical representations of your data

Documents -> splits -> embeddings -> vector store
then take question -> embed it -> find similar chunks -> pass that back to LLM for an answer



# Retrieval
Retrieve most relevant splits at query time


MMR - maximum marginal relevance
- fetch the most similar responses
- but choose the most diverse

LLM Aided Retreival
- self query
- parse queries -> split original query into filter and term
- can use metadata to filter on

Compression
- pull only most relevant bits from the retrieved documents

### Metadata - self-query retriever
- SelfQueryRetreiver uses an LLM to extract

### Compression
```
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
```

```
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectordb.as_retriever(search_type = "mmr")
)
```

Can combine this with MMR



# Question Answering
- question comes in
- look up relevant splits
- pass that in with the prompt to the LLM

Smaller context windows
- map_reduce
- refine
- map_rerank

# Chat - memory

Chat history

ConversationBufferMemory
ConversationalRetreivalChain

keeps conversation in a chat buffer to give it memory 


LEFT OFF 6:00 for the UI part


https://learn.deeplearning.ai/courses/langchain-chat-with-your-data/lesson/7/chat

