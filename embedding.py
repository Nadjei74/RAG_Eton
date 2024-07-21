#embedding.py
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
#ollama pull nomic-embed-text
def get_embedding_function():
    embeddings = OllamaEmbeddings(model="nomic-embed-text",show_progress=True)

    """embeddings = BedrockEmbeddings(
    credentials_profile_name = "default", region_name = "us-east-1"""""
    return embeddings