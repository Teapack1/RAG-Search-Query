import os
from dotenv import load_dotenv

load_dotenv()


from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext

documents = SimpleDirectoryReader("data").load_data()

print(documents)
index = VectorStoreIndex.from_documents(documents, show_progress=True)
print(index)

query_engine = index.as_query_engine()

from llama_index.retrievers import VectorIndexRetruver
from llama_index.query_engine import RetrieverQueryEngine

response = query_engine.query(
    "What flamability requirements do plastic enclosure have to meet?"
)
print(response)
