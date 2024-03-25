class ContextManager:
    _config = {
        'CHROMA_PATH': "chroma",
        'DATA_PATHS': ["Data/Excluded", "Data/Included"],
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
        'LOGGING_LEVEL': "INFO",
        'PROGRESS_BAR': True,
        'RESET_RESULTS': True,
        'CHECKPOINT_DICTIONARY': None,
        'PROMPT_TEMPLATE': None,
        'QUESTION': None,
        'CSV_HEADER': ["title", "converted", "embedded", "relevant", "checkpoints", "reasoning"]
    }

    @classmethod
    def get_config(cls, key):
        return cls._config.get(key)

    @classmethod
    def set_config(cls, key, value):
        cls._config[key] = value

