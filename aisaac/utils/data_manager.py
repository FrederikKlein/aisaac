import math
import os
import random

from langchain.document_loaders import DirectoryLoader

from aisaac.aisaac.utils.logger import Logger


class DocumentManager:
    def __init__(self, context_manager):
        self.data_paths = context_manager.get_config('DATA_PATHS')
        self.data_format = context_manager.get_config('DATA_FORMAT')
        self.random_subset = context_manager.get_config('RANDOM_SUBSET') == 'True'
        self.subset_size = int(context_manager.get_config('SUBSET_SIZE'))
        self.chroma_path = context_manager.get_config('CHROMA_PATH')
        self.global_data = []
        self.logger = Logger(__name__).get_logger()

    def load_data(self, path):
        loader = DirectoryLoader(path, glob=self.data_format)
        documents = loader.load()
        return documents

    def load_global_data(self):
        for data_path in self.data_paths:
            self.logger.info(f"Loading data from {data_path}.")
            data = self.load_data(data_path)
            self.global_data.append(data)

    def get_data(self):
        return_data = []
        len_data_sets = len(self.data_paths)
        if self.random_subset:
            for data_set in self.global_data:
                self.logger.info(f"Loading random subset of {self.subset_size / len_data_sets}.")
                return_data.extend(random.sample(data_set, int(self.subset_size / len_data_sets)))
        else:
            for data_set in self.global_data:
                return_data.extend(data_set)
        return return_data

    def __get_all_data(self):
        return_data = []
        for data_set in self.global_data:
            return_data.extend(data_set)
        return return_data

    def get_runnable_data(self):
        data = self.__get_all_data()
        for document in data[:]:
            title = os.path.splitext(os.path.basename(document.metadata["source"]))[0]
            if not os.path.exists(f"{self.chroma_path}/{title}"):
                self.logger.debug(f"Document store for {title} does not exist.")
                data.remove(document)
        if self.random_subset:
            data = random.sample(data, self.subset_size)
            self.logger.info(f"Loaded random runnable subset of {self.subset_size}.")
        else:
            self.logger.info(f"Loaded all runnable data.")
        return data


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.schema import Document


class VectorDataManager:

    def __init__(self, context_manager):
        self.apply_sentence_splitting_chunking = context_manager.get_config(
            'APPLY_SENTENCE_SPLITTING_CHUNKING') == 'True'
        self.chunk_size = int(context_manager.get_config('CHUNK_SIZE'))
        self.chunk_overlap = int(context_manager.get_config('CHUNK_OVERLAP'))
        self.model_manager = context_manager.get_model_manager()
        self.document_data_manager = context_manager.get_document_data_manager()
        self.system_manager = context_manager.get_system_manager()
        self.result_manager = context_manager.get_results_saver()
        self.chroma_path = self.system_manager.get_path("chroma")
        self.logger = Logger(__name__).get_logger()

        self.create_document_stores()

    def chunk_documents(self, documents):
        # switch between sentence splitting and recursive character splitting
        if self.apply_sentence_splitting_chunking:
            text_splitter = NLTKTextSplitter()
            chunks = text_splitter.split_text(documents)
            self.logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks using sentence splitting")
            return chunks

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        self.logger.info(f"Chunked {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    def save_to_chroma(self, chunks: list[Document], title: str, path: str):
        self.logger.info(f"Creating vector store for {title}.")
        self.logger.debug(f"Saving to {path}.")

        # system manager make directory
        self.system_manager.make_directory(path)

        # Create a new DB from the documents.
        try:
            db = Chroma.from_documents(
                chunks, self.model_manager.get_embedding(), persist_directory=path
            )
            db.persist()
            self.logger.debug(f"Saved {len(chunks)} chunks to {path}.")
        except Exception as e:
            self.logger.error(f"Error saving chunks to {path}: {e}")

    def create_document_stores(self):
        self.result_manager.reset_results()
        self.system_manager.reset_document_stores(self.chroma_path)
        data = self.document_data_manager.get_all_data()
        for document in data:
            title = os.path.splitext(os.path.basename(document.metadata["source"]))[0]
            path = f"{self.chroma_path}/{title}"
            # check if the document is already embedded
            if self.system_manager.path_exists(path):
                self.logger.info(f"Document store for {title} already exists and was not reset.")
                continue

            self.logger.debug(f"Creating document store for {title}.")
            self.result_manager.create_new_result_entry(title)

            try:
                self.save_to_chroma(self.chunk_documents([document]), title, path)
                self.result_manager.update_result_list(title, "embedded", True)
                self.logger.info(f"Created document store for {title}.")
            except:
                self.result_manager.update_result_list(title, "embedded", False)
                self.logger.error(f"Embedding failed for {title}.")

        self.logger.info("All document stores created.")

    def get_vectorstore(self, title: str):
        path = f"{self.chroma_path}/{title}"
        if not self.system_manager.path_exists(path):
            self.logger.error(f"Document store for {title} does not exist.")
            return None
        return Chroma(persist_directory=path, embdding_function=self.model_manager.get_embedding())

    def get_vectorstore_with_sigmoid_relevance_score_fn(self, title: str):
        path = f"{self.chroma_path}/{title}"
        if not self.system_manager.path_exists(path):
            self.logger.error(f"Document store for {title} does not exist.")
            return None
        return Chroma(
            persist_directory=path,
            embedding_function=self.model_manager.get_embedding(),
            collection_metadata={"hnsw:space": "l2"},
            relevance_score_fn=lambda distance: 1 / (1 + math.exp(-distance))
        )

    def get_vectorstores(self):
        vectorstores = {}
        for title in self.document_data_manager.get_all_titles():
            vectorstores[title] = self.get_vectorstore(title)
        return vectorstores

    def get_unified_vectorstore(self):
        vectorstores = self.get_vectorstores()
        vectorstore_baseline = Chroma(embedding_function=self.model_manager.get_embedding())
        for title in vectorstores:
            db2_data = vectorstores[title]._collection.get(include=['documents', 'metadatas', 'embeddings'])
            vectorstore_baseline._collection.add(
                embeddings=db2_data['embeddings'],
                metadatas=db2_data['metadatas'],
                documents=db2_data['documents'],
                ids=db2_data['ids']
            )
        return vectorstores
# %%
