from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from IPython.display import Markdown, display
from llama_index.core import Document, Settings
from llama_index.llms.openai import OpenAI

import chromadb

from dotenv import load_dotenv
import os

load_dotenv()  # This loads the variables from .env into the environment

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

Settings.llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

documents = SimpleDirectoryReader(
    "data/english"
).load_data()


db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("512_store")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

pipeline = IngestionPipeline(
    transformations=[
        SentenceSplitter(chunk_size=512, chunk_overlap=128),
        HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"),
    ],
    vector_store=vector_store,
)

# Ingest directly into a vector db
pipeline.run(documents=documents)


index = VectorStoreIndex.from_vector_store(vector_store)