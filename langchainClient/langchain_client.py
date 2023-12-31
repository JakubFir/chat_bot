from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from logger.logger_config import LOGGER

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 400


class LangchainClient:
    def load_data_from_youtube(self, url):
        try:
            LOGGER.info("starting loading data from the youtube url")
            loader = YoutubeLoader.from_youtube_url(url)
            LOGGER.info("successfully loaded data from youtube")
            return loader.load()
        except Exception as e:
            LOGGER.exception(f"An error occurred while trying to load data from youtube: {e}")
            raise

    def split_document(self, document):
        LOGGER.info("starting splitting the document")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=400)
        return text_splitter.split_documents(document)

    def initialize_qa_chain(self, llm):
        LOGGER.info("starting qa chain")
        return load_qa_chain(llm=llm, chain_type="stuff")

    def make_a_query(self, query, docsearch, chain):
        LOGGER.info(f"starting searching for similarity in docs for query: {query}")
        docs = docsearch.similarity_search(query)
        prompt_template = PromptTemplate.from_template(
            f"The only thing you know is the provided contex of video talk."
            f" Based on the context, answer as truthfully as you can to this: {query}."
            f"If the question is not related to context answer: I dont know"
        )
        print(chain.run(input_documents=docs, question=prompt_template.template))


