import os
from dotenv import load_dotenv
import chromadb
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from IPython.display import Markdown, display
from llama_index.llms.openai import OpenAI


load_dotenv()
Settings.llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("512_store")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(
    vector_store
)

query_engine = index.as_query_engine(
    settings=Settings
)

response = query_engine.query("What cable do I use to hang a 1.5kg heavy luminaire on?")

print(f"Response: {response}")