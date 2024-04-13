# ContextManager Class Documentation

## Overview
The `ContextManager` class in the `aisaac` package is responsible for managing the configuration and facilitating the integration of various utility modules such as data handling, model management, and result saving within an application context. This class uses a configuration dictionary to customize the behavior of these utilities according to specific application needs.

## Configuration Details
The `ContextManager` initializes with a comprehensive set of default configurations, which can be overridden by user-provided configurations upon initialization. The configuration parameters include paths, model details, data handling options, logging settings, and various operational flags. Each parameter serves a specific function within the system, impacting how data is processed, stored, and how models are managed. Below is an elaboration on some key configuration parameters:
#### Checkpoint Dictionary
- **`CHECKPOINT_DICTIONARY`**: Mapping of conditions to boolean evaluations for decision processes.

#### Path Settings
- **`BASE_DIR`**: Base directory of the application for relative paths.
- **`CHROMA_PATH`**: Directory for chroma-related files.
- **`DATA_PATHS`**: Paths for loading data, useful for segregating included and excluded data.
- **`BIN_PATH`**: Base directory for binary data storage.
- **`RESULT_PATH`**: Directory for saving result files.
- **`ORIGINAL_RESULT_PATH`**: Path for storing original or gold standard results.
- **`RESULT_FILE`**: Name of the result file.
- **`ORIGINAL_RESULT_FILE`**: Name of the file for original results.
> Make sure that the paths exist and are correctly set to avoid errors during operations.

#### Model Management
- **`MODEL_CLIENT_URL`**: URL for interacting with hosted machine learning models. Ollama models are hosted at "http://localhost:11434" by default
- **`EMBEDDING_MODEL`**: Identifier for the text embedding model.
- **`RAG_MODEL`**: Identifier for the Retrieve-And-Generate model.
- **`LOCAL_MODELS`**: Whether models are hosted locally (True) or remotely (False).

#### Data Processing
- **`DATA_FORMAT`**: Expected file format for input data, such as "*.pdf".
- **`RANDOM_SUBSET`**: Whether to use a random subset of data, typically for testing.
- **`SUBSET_SIZE`**: Size of the data subset if `RANDOM_SUBSET` is true.
- **`CHUNK_SIZE`**: Size of data chunks for processing.
- **`CHUNK_OVERLAP`**: Overlap size between data chunks.

#### Similarity Search
- **`RELEVANCE_THRESHOLD_CUTOFF`**: Cutoff threshold for relevance scoring.
- **`APPLY_RELEVANCE_THRESHOLD`**: Whether to apply the relevance threshold.
- **`SIMILARITY_SEARCH_K`**: Number of nearest neighbors to retrieve in similarity searches.

#### Criteria Optimization
- **`FEATURE_IMPORTANCE_THRESHOLD`**: Threshold how important a feature has to be to be optimized.
- **`IMPORTANCE_GREATER_THAN_THRESHOLD`**: Whether the feature has to be more (True) or less (False) important the the `FEATURE_IMPORTANCE_THRESHOLD`. Default of True
- **`MAX_FEATURE_IMPROVEMENT_DOCUMENTS`**: Max number of documents to consider for feature improvements in context-sensitive optimization techniques.
- **`NUMBER_EXPERT_CHOICES`**: Number of choices presented to the expert in human-in-the-loop optimization techniques.

#### Logging and Debugging
- **`LOGGING_LEVEL`**: Severity level of logs (e.g., DEBUG, INFO).
- **`VERBOSE_CODE`**: Toggle for verbose logging. Prints additional information during operations.
- **`RESET_RESULTS`**: Whether to clear previous results before new operations.
- **`PROGRESS_BAR`**: Display of progress bar during operations. (TODO not functional)

#### Please don't touch these right now :)
- **`CSV_HEADER`**: Header row for CSV output files.
- **`PROMPT_TEMPLATE`**: Template for constructing prompts in interactive scenarios.
- **`QUESTION`**: A question string for user interactions or queries.
- **`APPLY_RERANKING`**: Whether reranking of results is enabled. Note: This needs a working cohere account. Please leave it as False.




## Methods
- `__init__(self, config=None)`: Initializes a new context manager instance. If a configuration dictionary is provided, it merges with the default configuration, with the user's settings taking precedence.
- `get_config(self, key)`: Retrieves the value for the specified configuration key.
- `set_config(self, key, value)`: Sets or updates the configuration value for the specified key.
- `get_vector_data_manager(self)`: Returns an instance of `VectorDataManager` configured for this context.
- `get_document_data_manager(self)`: Returns an instance of `DocumentManager` configured for this context.
- `get_result_saver(self)`: Returns an instance of `ResultSaver` configured for this context.
- `get_model_manager(self)`: Returns an instance of `ModelManager` configured for this context.
- `get_system_manager(self)`: Returns an instance of `SystemManager` configured for this context.
- `get_similarity_searcher(self)`: Returns an instance of `SimilaritySearcher` configured for this context.
- `get_logger(self)`: Returns a `Logger` instance configured for this context.

## Usage Example
```python
# Creating a ContextManager instance with custom configurations
custom_config = {
    'DATA_PATHS': ["Data/NewData"],
    'VERBOSE_CODE': False
}
context_manager = ContextManager(config=custom_config)

# Accessing a model manager from the context
model_manager = context_manager.get_model_manager()
```

```python
# Creating a default ContextManager instance and changing the configurations later
context_manager = ContextManager()
context_manager.set_config('DATA_PATHS', ["Data/NewData"])
context_manager.set_config('VERBOSE_CODE', False)

# Accessing a model manager from the context
model_manager = context_manager.get_model_manager()
```

## Note
The `ContextManager` plays a pivotal role in integrating various components and managing the flow of data and operations within the system, making it crucial for users to understand and configure it properly to suit their specific needs.