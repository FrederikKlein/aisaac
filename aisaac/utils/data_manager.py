import math
import os
import pickle
import random
from collections import Counter

from langchain.schema import Document
from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

from aisaac.aisaac.utils.logger import Logger


class DocumentManager:
    def __init__(self, context_manager):
        self.system_manager = context_manager.get_system_manager()
        relative_data_paths = context_manager.get_config('DATA_PATHS')
        self.data_paths = self.system_manager.get_full_paths(relative_data_paths)
        self.data_format = context_manager.get_config('DATA_FORMAT')
        self.random_subset = context_manager.get_config('RANDOM_SUBSET') == 'True'
        self.subset_size = int(context_manager.get_config('SUBSET_SIZE'))
        self.chroma_path = context_manager.get_config('CHROMA_PATH')
        self.full_chroma_path = self.system_manager.get_full_path(self.chroma_path)
        self.bin_path = self.system_manager.get_full_path(context_manager.get_config('BIN_PATH'))
        self.system_manager.make_directory(context_manager.get_config('BIN_PATH'))
        self.global_data = []
        self.logger = Logger(__name__).get_logger()

    def __load_data(self, path):
        try:
            if self.data_format == '*.pdf':
                loader = PyPDFLoader(path)
                documents = loader.load()
                documents = self.clean_and_join_document_pages(documents)
            elif self.data_format == '*.md':
                loader = UnstructuredMarkdownLoader(path)
                documents = loader.load()
            else:
                self.logger.error(f"Data format {self.data_format} not supported.")
                return []
            return documents
        except Exception as e:
            self.logger.error(f"Error loading data from {path}: {e}")
            return []

    def update_global_data(self):
        # remove any data from global data that isn't in the data paths
        for data_set in self.global_data:
            for data in data_set:
                if all(data.metadata["source"] not in data_path for data_path in self.data_paths):
                    self.global_data.remove(data_set)
                    break
        for data_path in self.data_paths:
            for document_path in os.listdir(data_path):
                relative_document_path = f"{data_path}/{document_path}"
                # TODO check if the data is already in the global data

                self.logger.info(f"Loading data from {relative_document_path}.")
                data = self.__load_data(relative_document_path)
                if data is not None:
                    self.global_data.append(data)
                else:
                    self.logger.error(f"Error loading data from {relative_document_path}.")

        self.__save_global_data()

    def get_data(self):
        return_data = []
        self.__load_global_data()
        # self.update_global_data()
        len_data_sets = len(self.data_paths)
        if self.random_subset:
            for data_set in self.global_data:
                self.logger.info(f"Loading random subset of {self.subset_size / len_data_sets}.")
                return_data.extend(random.sample(data_set, int(self.subset_size / len_data_sets)))
        else:
            for data_set in self.global_data:
                return_data.extend(data_set)
        return return_data

    def get_all_titles(self):
        return_titles = []
        for data_path in self.data_paths:
            for document_path in os.listdir(data_path):
                # get title from data_path
                title = os.path.basename(document_path)
                return_titles.append(title)
        return return_titles

    def get_converted_titles(self):
        return_titles = []
        all_titles = self.get_all_titles()
        global_data = self.__load_global_data()
        # compare the titles in the global data with the all titles
        for data_set in global_data:
            for data in data_set:
                title = self.system_manager.get_title_without_extension(data.metadata["source"])
                if title in all_titles:
                    return_titles.append(title)
        return return_titles


    def get_runnable_titles(self):
        global_titles = self.get_all_titles()
        for title in global_titles[:]:
            # remove end of file extension
            dir_title = os.path.splitext(title)[0]
            self.logger.debug(f"Checking {self.chroma_path}/{dir_title}")
            if not self.system_manager.path_exists(f"{self.chroma_path}/{dir_title}"):
                self.logger.debug(f"Document store for {dir_title} does not exist.")
                global_titles.remove(title)
            else:
                self.logger.debug(f"Document store for {dir_title} exists.")
        return global_titles

    def __get_all_data(self):
        return_data = []
        self.__load_global_data()
        for data_set in self.global_data:
            return_data.extend(data_set)
        return return_data

    def get_runnable_data(self):
        data = self.__get_all_data()
        for document in data[:]:
            title = os.path.splitext(os.path.basename(document.metadata["source"]))[0]
            if not self.system_manager.path_exists(f"{self.chroma_path}/{title}"):
                self.logger.debug(f"Document store for {title} does not exist.")
                data.remove(document)
        if self.random_subset:
            data = random.sample(data, self.subset_size)
            self.logger.info(f"Loaded random runnable subset of {self.subset_size}.")
        else:
            self.logger.info(f"Loaded all runnable data.")
        return data

    def clean_and_join_document_pages(self, documents):
        # Return the input directly if there's only one page or no pages at all
        if len(documents) <= 1:
            return documents

        # Assuming docs is a list of Document objects
        lines_across_pages = [doc.page_content.split('\n') for doc in documents]

        # Flatten the list to count occurrences of each line
        all_lines = [line for page in lines_across_pages for line in page]
        line_counts = Counter(all_lines)

        # Assume a header or footer is repeated if it appears on more than half of the pages
        repetition_threshold = len(documents) // 2
        repeated_lines = {line for line, count in line_counts.items() if count > repetition_threshold}

        # Filter out repeated lines from each page
        cleaned_pages = []
        for page in lines_across_pages:
            cleaned_page = '\n'.join(line for line in page if line not in repeated_lines)
            cleaned_pages.append(cleaned_page)

        # Combine cleaned pages into one string
        combined_content = "\n".join(cleaned_pages)

        combined_metadata = documents[0].metadata

        # remove the metadata [page] from the combined metadata, keeping the rest of it
        combined_metadata.pop('page', None)
        combined_document = [Document(page_content=combined_content, metadata=combined_metadata)]
        return combined_document

    def __save_global_data(self):
        # Serialize and save self.global_data to a binary file
        with open(f'{self.bin_path}/doc_mngr.pkl', 'wb') as f:
            pickle.dump(self.global_data, f)

    def __load_global_data(self):
        # Serialize and save self.global_data to a binary file
        with open(f'{self.bin_path}/doc_mngr.pkl', 'rb') as f:
            self.global_data = pickle.load(f)


class VectorDataManager:

    def __init__(self, context_manager):
        self.apply_sentence_splitting_chunking = context_manager.get_config(
            'APPLY_SENTENCE_SPLITTING_CHUNKING') == 'True'
        self.chunk_size = int(context_manager.get_config('CHUNK_SIZE'))
        self.chunk_overlap = int(context_manager.get_config('CHUNK_OVERLAP'))
        self.model_manager = context_manager.get_model_manager()
        self.document_data_manager = context_manager.get_document_data_manager()
        self.system_manager = context_manager.get_system_manager()
        self.result_manager = context_manager.get_result_saver()
        self.chroma_path = context_manager.get_config('CHROMA_PATH')
        self.full_chroma_path = self.system_manager.get_full_path(self.chroma_path)
        self.logger = Logger(__name__).get_logger()

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
        self.system_manager.reset_directory(self.chroma_path)
        data = self.document_data_manager.get_data()
        for document in data:
            title = self.system_manager.get_title_without_extension(document.metadata["source"])
            path = f"{self.chroma_path}/{title}"
            # check if the document is already embedded
            if self.system_manager.path_exists(path):
                self.logger.info(f"Document store for {title} already exists and was not reset.")
                continue

            self.logger.debug(f"Creating document store for {title}.")
            self.result_manager.create_new_result_entry(title)
            self.result_manager.update_result_list(title, "converted", True)

            try:
                chunks = self.chunk_documents([document])
                self.save_to_chroma(chunks, title, path)
                self.result_manager.update_result_list(title, "embedded", True)
                self.logger.info(f"Created document store for {title}.")
            except Exception as e:
                self.result_manager.update_result_list(title, "embedded", False)
                self.logger.error(f"Embedding failed for {title}.")

        self.logger.info("All document stores created.")

    def get_vectorstore(self, title: str):
        title = os.path.splitext(title)[0]
        path = f"{self.chroma_path}/{title}"
        if not self.system_manager.path_exists(path):
            self.logger.error(f"Document store for {title} does not exist.")
            return None
        return Chroma(persist_directory=self.system_manager.get_full_path(path),
                      embedding_function=self.model_manager.get_embedding()
                      )

    def get_vectorstore_with_sigmoid_relevance_score_fn(self, title: str):
        title = os.path.splitext(title)[0]
        path = f"{self.chroma_path}/{title}"
        if not self.system_manager.path_exists(path):
            self.logger.error(f"Document store for {title} does not exist.")
            return None
        return Chroma(
            persist_directory=self.system_manager.get_full_path(path),
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
            db2_data = vectorstores[title]._collection.get(include=['documents', 'metadata', 'embeddings'])
            vectorstore_baseline._collection.add(
                embeddings=db2_data['embeddings'],
                metadatas=db2_data['metadata'],
                documents=db2_data['documents'],
                ids=db2_data['ids']
            )
        return vectorstores
# %%
