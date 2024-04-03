from aisaac.aisaac.utils import DocumentManager
from aisaac.aisaac.utils import Logger
from aisaac.aisaac.utils import ModelManager
from aisaac.aisaac.utils import ResultSaver
from aisaac.aisaac.utils import SimilaritySearcher
from aisaac.aisaac.utils import SystemManager
from aisaac.aisaac.utils import VectorDataManager


class ContextManager:
    _default_config = {
        'CHROMA_PATH': "chroma",
        'DATA_PATHS': ["Data/Excluded", "Data/Included"],
        'BIN_PATH': 'bin',
        'RESULT_PATH': "results",
        'RESULT_FILE': "results.csv",
        'MODEL_CLIENT_URL': "http://localhost:11434",
        'EMBEDDING_MODEL': "nomic-embed-text:latest",
        'RAG_MODEL': "mixtral:latest",
        'LOCAL_MODELS': True,
        'DATA_FORMAT': "*.pdf",
        'RANDOM_SUBSET': False,
        'SUBSET_SIZE': 5,
        'RELEVANCE_THRESHOLD_CUTOFF': 0.7,
        'APPLY_RELEVANCE_THRESHOLD': True,
        'APPLY_RERANKING': False,
        'SIMILARITY_SEARCH_K': 4,
        'CHUNK_SIZE': 1000,
        'CHUNK_OVERLAP': 100,
        'VERBOSE_CODE': True,
        'LOGGING_LEVEL': "DEBUG",
        'PROGRESS_BAR': True,
        'RESET_RESULTS': True,
        'BASE_DIR': 'aisaac',
        'PROMPT_TEMPLATE': None,
        'QUESTION': None,
        'CSV_HEADER': ["title", "converted", "embedded", "relevant", "checkpoints", "reasoning"],
        'CHECKPOINT_DICTIONARY': {"Thyroid Cancer Types": "If the study involves any type of thyroid cancer such as "
                                                          "Papillary TC, Follicular TC, Medullary TC, "
                                                          "Poorly Differentiated TC, Anaplastic TC, Hurtle cell "
                                                          "carcinoma, or Non-invasive follicular thyroid neoplasm "
                                                          "with papillary-like nuclear features (NIFTP), then return "
                                                          "True. However, If the study involves non-malignant "
                                                          "entities or conditions other than the specified types of "
                                                          "thyroid cancer, then return False.",

                                  "Study Population": "If the study is conducted on human subjects, then return True. "
                                                      "Otherwise, if the study is conducted on an organism different "
                                                      "than human,  animals or uses cell lines,  return False."},
        'FEATURE_IMPORTANCE_THRESHOLD': 0.1,
        'MAX_FEATURE_IMPROVEMENT_DOCUMENTS': 20,
        'NUMBER_EXPERT_CHOICES': 3,
        'IMPORTANCE_GREATER_THAN_THRESHOLD': True,
    }

    def __init__(self, config=None):
        """
        Initialize a new context with the given configuration, falling back on defaults where necessary.

        :param config: A dictionary containing configuration settings.
        """
        # Merge user-provided config with defaults. User config overrides defaults.
        self._config = {**self._default_config, **(config or {})}

    def get_config(self, key):
        """
        Get a configuration value.

        :param key: The configuration key to retrieve.
        :return: The configuration value.
        """
        return self._config.get(key)

    # this might have to be deleted in order to avoid logic issues
    def set_config(self, key, value):
        """
        Set a configuration value.

        :param key: The configuration key to set.
        :param value: The new value for the key.
        """
        self._config[key] = value

    def get_vector_data_manager(self):
        """
        Get an instance of VectorDataManager for this context.
        """
        return VectorDataManager(self)

    def get_document_data_manager(self):
        """
        Get an instance of DocumentManager for this context.
        """
        return DocumentManager(self)

    def get_result_saver(self):
        """
        Get an instance of ResultSaver for this context.
        """
        return ResultSaver(self)

    def get_model_manager(self):
        """
        Get an instance of ModelManager for this context.
        """
        return ModelManager(self)

    def get_system_manager(self):
        """
        Get an instance of SystemManager for this context.
        """
        return SystemManager(self)

    def get_similarity_searcher(self):
        """
        Get an instance of SimilaritySearcher for this context.
        """
        return SimilaritySearcher(self)

    def get_logger(self):
        """
        Get a Logger instance for this context.
        """
        return Logger(__name__, self)


#%%
