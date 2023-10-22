from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from logger.logger_config import LOGGER
from env_variables.env_variables import load_configuration

openai_api_key = load_configuration()


class OpenAI:

    def initialize_openai(self):
        LOGGER.info("starting initialization of openaiClient")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo",
                          openai_api_key=openai_api_key[0])

    def initialize_embeddings(self):
        LOGGER.info("starting initialization of embedding")
        return OpenAIEmbeddings(openai_api_key=openai_api_key[0])
