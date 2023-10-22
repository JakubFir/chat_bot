import pinecone
from langchain.vectorstores.pinecone import Pinecone
from logger.logger_config import LOGGER
from env_variables.env_variables import load_configuration

pinecone_api_key = load_configuration()


class PineconeClient:
    def initialize_pinecone(self):
        LOGGER.info("starting initialization of pineconeClient")
        pinecone.init(
            api_key=pinecone_api_key[1],
            environment="gcp-starter")

    def initialize_pinecone_search(self, texts, embeddings, index_name):
        LOGGER.info("starting retrieving data from pineconeClient")
        if self.validate_vectors(texts, index_name):
            docsearch = Pinecone.from_texts(
                [doc.page_content for doc in texts],
                embedding=embeddings,
                index_name=index_name,
            )
            return docsearch
        else:
            docsearch = Pinecone.from_existing_index(index_name, embeddings)
        return docsearch

    def validate_vectors(self, texts, index_name):
        vector_count = 0
        stats = Pinecone.get_pinecone_index(index_name).describe_index_stats()
        namespace_map = stats['namespaces']
        for namespace in namespace_map:
            vector_count = namespace_map[namespace]['vector_count']
        if not vector_count >= len(texts):
            return True
        else:
            return False
