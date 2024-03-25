import os
import random

from langchain.document_loaders import DirectoryLoader

from aisaac.aisaac.utils.logger import Logger


class DataManager:
    def __init__(self, context_manager):
        self.context_manager = context_manager
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
