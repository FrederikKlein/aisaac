from time import sleep

import ollama
from langchain.embeddings import XinferenceEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.llms.xinference import Xinference
from xinference.client import Client

from aisaac.aisaac.utils.logger import Logger


class ModelManager:
    def __init__(self, context_manager):
        self.use_local_models = str(context_manager.get_config('LOCAL_MODELS')).lower() == 'true'
        self.model_client_url = context_manager.get_config('MODEL_CLIENT_URL')
        self.embedding_model_id = context_manager.get_config('EMBEDDING_MODEL')
        self.rag_model_id = context_manager.get_config('RAG_MODEL')
        self.my_chat_model, self.embedding, self.model_uids = None, None, None
        self.logger = Logger(__name__).get_logger()

        self.__set_up_models()

    def __set_up_embedding(self):
        if self.use_local_models:
            embedding_setup = OllamaEmbeddings(base_url=self.model_client_url, model=self.embedding_model_id)
        else:
            embedding_setup = XinferenceEmbeddings(
                server_url=self.model_client_url,
                model_uid=self.model_uids[self.embedding_model_id])
        return embedding_setup

    def __set_up_rag_model(self):
        if self.use_local_models:
            rag_model_setup = Ollama(base_url=self.model_client_url, model=self.rag_model_id)
        else:
            rag_model_setup = Xinference(
                server_url=self.model_client_url,
                model_uid=self.model_uids[self.rag_model_id]
            )
        return rag_model_setup

    def __set_up_models(self):
        if self.use_local_models:
            self.logger.info("Using local models.")
            models = ollama.list()
            model_uids = {model['name']: model['digest'] for model in models["models"]}
            self.logger.info(f"Available models: {list(model_uids.keys())}")

            if self.rag_model_id not in model_uids:
                self.logger.critical(f"{self.rag_model_id} does not exist")
            else:
                self.my_chat_model = self.__set_up_rag_model()

            if self.embedding_model_id not in model_uids:
                self.logger.critical(f"{self.embedding_model_id} does not exist")
            else:
                self.embedding = self.__set_up_embedding()

        else:
            self.logger.info("Using cosy models.")
            cosy_client = Client(self.model_client_url)
            # cosyClient.login("student", "students_key")
            models = cosy_client.list_models()
            model_uids = {model['model_name']: uid for uid, model in models.items()}
            self.logger.info(f"Available models: {list(model_uids.keys())}")

            if self.embedding_model_id not in model_uids:
                self.logger.critical(f"{self.embedding_model_id} does not exist")
            else:
                self.embedding = self.__set_up_embedding()

            if self.rag_model_id not in model_uids:
                self.logger.critical(f"{self.rag_model_id} does not exist")
            else:
                self.my_chat_model = self.__set_up_rag_model()

    def get_embedding(self):
        while not self.embedding:
            sleeping_time = 5
            self.logger.critical(
                f"{self.embedding_model_id} does not exist. Please check the connection and the model name. "
                f"Trying again in {sleeping_time} seconds.")
            sleep(sleeping_time)
            self.logger.info(f"Trying to get {self.embedding_model_id}.")
            self.embedding = self.__set_up_embedding()
            sleeping_time *= 2
        return self.embedding

    def get_rag_model(self):
        while not self.my_chat_model:
            sleeping_time = 5
            self.logger.critical(
                f"{self.rag_model_id} does not exist. Please check the connection and the model name. "
                f"Trying again in {sleeping_time} seconds.")
            sleep(sleeping_time)
            self.logger.info(f"Trying to get {self.rag_model_id}.")
            self.my_chat_model = self.__set_up_rag_model()
            sleeping_time *= 2
        return self.my_chat_model

# %%
