import warnings

import cohere
from langchain_core.documents import Document

from aisaac.aisaac.utils.logger import Logger


class SimilaritySearcher:
    def __init__(self, context_manager):
        # Fetch all necessary configurations from the context manager
        self.system_manager = context_manager.get_system_manager()
        self.vector_data_manager = context_manager.get_vector_data_manager()
        self.logger = Logger(__name__).get_logger()
        self.similarity_search_k = int(context_manager.get_config('SIMILARITY_SEARCH_K'))
        self.relevance_threshold = float(context_manager.get_config('RELEVANCE_THRESHOLD_CUTOFF'))
        self.apply_reranking = context_manager.get_config('RERANKING') == 'True'
        self.apply_relevance_threshold = context_manager.get_config('RELEVANCE_THRESHOLD') == 'True'

    def __apply_reranking_method(self, results, query_text):
        # TODO replace with actual API key
        co = cohere.Client("Dz03PKYzHRwgvTgPWucDClzr2RylARxhmex7cGJ9")
        # the content of results is docs
        docs = [doc.page_content for doc, _score in results]
        # TODO replace with actual model
        rerank_hits = co.rerank(query=query_text, documents=docs, top_n=3, model='rerank-multilingual-v2.0')
        # handle return format. Can be found here: https://docs.cohere.com/reference/rerank-1
        formatted_hits = [(Document(page_content=result.document['text']), result.relevance_score) for result in
                          rerank_hits]
        return formatted_hits

    def __apply_relevance_threshold_method(self, results):
        return [result for result in results if result[1] > self.relevance_threshold]

    def similarity_search(self, document_title, query_text):
        db = self.vector_data_manager.get_vectorstore(document_title)
        # debugging
        self.logger.debug(f"This is the collection count: {db._collection.count()}")
        self.logger.debug(f"This is the collection: {db._collection}")
        self.logger.debug(f"And this is the result for the sim_search: {db.similarity_search(query_text, k=self.similarity_search_k)}")


        self.logger.debug(f"Conducting similarity search for {document_title} with following query: \n{query_text}.")
        # catching warnings from the similarity search and trying again with a different relevance score function
        with warnings.catch_warnings(record=True) as caught_warnings:
            results = db.similarity_search_with_relevance_scores(query_text, k=self.similarity_search_k)
            for _ in caught_warnings:
                # This relevance score function is a sigmoid function
                db = self.vector_data_manager.get_vectorstore_with_sigmoid_relevance_score_fn(document_title)
                with warnings.catch_warnings(record=True) as more_caught_warnings:
                    results = db.similarity_search_with_relevance_scores(query_text, k=self.similarity_search_k)
                    for _ in more_caught_warnings:
                        # Here we don't generate a score, but add a default in the end
                        results = db.similarity_search(query_text, k=self.similarity_search_k)
                        results = [(document, self.relevance_threshold) for document in results]
                        self.logger.warning(
                            f"Could not find a relevance score function that works for {document_title}."
                            f"Applied default relevance threshold of {self.relevance_threshold}.")
        self.logger.debug(f"Found {len(results)} results.")
        if self.apply_reranking:
            results = self.__apply_reranking_method(results, query_text)
            self.logger.info(f"Applied Reranking. {len(results)} results remain.")
        else:
            self.logger.info("Not applying Reranking.")
        if self.apply_relevance_threshold:
            results = self.__apply_relevance_threshold_method(results)
            self.logger.info(f"Applied relevance threshold. {len(results)} results remain.")
        else:
            self.logger.debug("Not applying relevance threshold.")

        return results
