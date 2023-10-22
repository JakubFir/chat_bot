from langchain.prompts import PromptTemplate
from unittest import TestCase

from langchainClient.langchain_client import LangchainClient
from unittest.mock import Mock, patch


class Test(TestCase):

    def setUp(self):
        self.client = LangchainClient()

    def test_make_a_query(self):
        query = "What is the capital of France?"
        docsearch = Mock()
        chain = Mock()

        mock_similarity_search_result = [Mock(), Mock()]
        docsearch.similarity_search.return_value = mock_similarity_search_result

        mock_chain_result = "Paris"
        chain.run.return_value = mock_chain_result

        prompt_template = PromptTemplate.from_template(
            f"The only thing you know is the provided contex."
            f" Using the contex of the video talk answer to this: {query}"
            f" If you cant find answer in the contex, replay : I'm only allowed to answer question about video"
        )
        LangchainClient.make_a_query(self, query, docsearch, chain)

        docsearch.similarity_search.assert_called_with(query)

        chain.run.assert_called_with(input_documents=mock_similarity_search_result, question=prompt_template.template)
