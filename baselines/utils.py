from typing import Union
import uuid
import numpy as np

def retrieve(query: str, metadata: bool = False) -> dict[str, Union[list[str], list[dict]]]:
    """
    Simulates retrieving the most relevant text from a vector database, given any input text. Flag metadata = True if you want to also retrieve the metadata.
    Returns dict in form {"text": <LIST OF TEXTS>, "metadata": <LIST OF METADATA>}. 
    """
    return {"text": [""], "metadata": [{}]}

def insert(vector: np.array, metadata: dict = {}) -> str:
    """
    Simulates inserting a given vector and metadata to a vector database. Returns a string representing its ID in the database. 
    """
    return str(uuid.uuid4())

def generate(query: str) -> str:
    """
    Simulates generating an LLM output given a query. Returns the string result of the LLM call.
    """
    return "As a large language model..."

def embed(text: str) -> np.array:
    """
    Simulates embedding a given input text to a vector array.
    """
    return np.array([0])

def chunk(text: str) -> list[str]:
    """
    Simulates splitting up a body of text into a list of chunks.
    """
    return [""]
