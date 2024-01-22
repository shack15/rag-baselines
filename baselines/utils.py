from dotenv import load_dotenv
from typing import Union
import uuid
import numpy as np
from openai import OpenAI
import pyembeddings
import os

load_dotenv()


# Mock implementation of the services needed to build this module
pyembeddings.init(os.getenv('PYEMBEDDINGS_KEY'))

# EMBEDDINGS GENERATION
gen = pyembeddings.Generator()

# VECTOR DATABASE
db = pyembeddings.Database()
# create a collection called rag_collection that uses the BAAI/bge-base-en-v1.5 embedding model
collection = db.create_collection("rag_collection", 'BAAI/bge-base-en-v1.5')

# LLM
client = OpenAI(API_KEY = os.getenv('OPENAI_API_KEY'))

def retrieve(vector: np.array, metadata: bool = False) -> dict[str, Union[list[str], list[dict]]]:
    """
    Simulates retrieving the most relevant text from a vector database, given any input vector. Flag metadata = True if you want to also retrieve the metadata.
    Returns dict in form {"text": <LIST OF TEXTS>, "metadata": <LIST OF METADATA>}. 
    """

    ##### EXAMPLE IMPLEMENTATION #####
    relevant_texts = collection.query(query_embedding = vector)
    text = relevant_texts['documents'][0]
    metadata = relevant_texts['metadata'][0]
    return {"text": text, "metadata": metadata}
    ##################################

    # return {"text": [""], "metadata": [{}]}

def insert(vector: np.array, metadata: dict = {}) -> str:
    """
    Simulates inserting a given vector and metadata to a vector database. Returns a string representing its ID in the database. 
    """
    db.insert(vector, metadata)
    return str(uuid.uuid4())

def generate(query: str) -> str:
    """
    Simulates generating an LLM output given a query. Returns the string result of the LLM call.
    """

    ##### EXAMPLE IMPLEMENTATION #####
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query}
        ]
        )
    ##################################
    
    # return "As a large language model..."

def embed(text: str) -> np.array:
    """
    Simulates embedding a given input text to a vector array.
    """
    ##### EXAMPLE IMPLEMENTATION #####
    embedding = gen.embed([text])
    return embedding[0]
    ##################################

    # return np.array([0])
def chunk(text: str) -> list[str]:
    """
    Simulates splitting up a body of text into a list of chunks.
    """
    return [""]
