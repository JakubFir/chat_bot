import os
from dotenv import find_dotenv, load_dotenv

def load_configuration():
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    openai_api_key = os.getenv("openai_api_key")
    pinecone_api_key = os.getenv("pinecone_api_key")
    return openai_api_key, pinecone_api_key


