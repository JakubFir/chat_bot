from openaiClient.openai_client import OpenAI
from langchainClient.langchain_client import LangchainClient
from pineconeClient.pinecone_client import PineconeClient


def initialize_start():
    openai_client = OpenAI()
    langchain_client = LangchainClient()
    pinecone_client = PineconeClient()
    document = langchain_client.load_data_from_youtube("https://www.youtube.com/watch?v=W3S0SzYWKvA")
    documents = langchain_client.split_document(document)
    embeddings = openai_client.initialize_embeddings()
    pinecone_client.initialize_pinecone()
    docsearch = pinecone_client.initialize_pinecone_search(documents, embeddings, "chat")
    llm = openai_client.initialize_openai()
    chain = langchain_client.initialize_qa_chain(llm)
    start(docsearch, chain, langchain_client)


def start(docsearch, chain, langchain_client):
    while True:
        print("Provide a question (or type 'quit' to exit):")
        query = input()
        if query == "quit":
            break
        langchain_client.make_a_query(query, docsearch, chain)
