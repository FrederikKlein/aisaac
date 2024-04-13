![aisaac logo v2](https://github.com/FrederikKlein/aisaac/assets/94715827/90a82ee5-9b58-4839-ac48-01d82123eab9)



# aisaac
> **🔮** *An Intelligent Screening Assistant for Academic Content*

aisaac is a tool designed to assist researchers and academics in screening and evaluating academic papers. By defining custom criteria, users can efficiently sift through large volumes of academic content to find papers that are most relevant to their research. Additionally, aisaac offers capabilities to evaluate its performance and optimize criteria based on user-provided ground truth.

## Features
- **Document Reading**: Automatically reads and preprocesses academic papers.
- **Criteria Evaluation**: Evaluates papers against user-defined criteria to determine relevance.
- **Performance Evaluation**: Assesses the tool's performance using provided ground truth.
- **Criteria Optimization**: Improves criteria based on performance evaluation for more accurate future screenings.
- **User Interface**: Offers a user-friendly interface for interaction with the tool.

## Installation

Before installing `aisaac`, ensure you have a compatible Python version installed. Python >= 3.9 is required. The system was tested on Python 3.10 though.
Then, follow these steps:

1. Clone the `aisaac` repository:
   ```bash
   git clone https://github.com/FrederikKlein/aisaac.git
    ```
2. Navigate to the project directory:
    ```bash
    cd aisaac
    ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
`aisaac` can be used as a python package. 
<!--- 
or through its graphical user interface (GUI) 
--->

### Python Package Quickstart Guide
To use `aisaac` as a python package, import the `aisaac` module and use its classes and functions as needed. 
The core classes are
- Screener
- Evaluator
- CriteriaOptimizer

To use any of these classes, first import them:
```python
from aisaac.aisaac.core.screener import Screener
from aisaac.aisaac.core.evaluator import Evaluator
from aisaac.aisaac.core.criteria_optimizer import CriteriaOptimizer
```

All configurations and contexts are handled by the ContextManager class. This class is part of the utils module and can be imported as follows:
```python
from aisaac.aisaac.utils.context_manager import ContextManager
```

The core classes use a context manager which is filled with default values. The context manager can be used to set custom values for the classes.

Here is how to set custom values:
```python
cm = ContextManager()
cm.set_config('RAG_MODEL', "mixtral:latest")
cm.set_config('RESULT_FILE', "results.csv")
cm.set_config('CHECKPOINT_DICTIONARY', checkpoint_dict)
```

To connect to a Server, use the following code snippet:
```python
cm.set_config('LOCAL_MODELS', "False")
cm.set_config('MODEL_CLIENT_URL', "https://llm.cosy.bio")
cm.set_config('EMBEDDING_MODEL', "gte-large")
cm.set_config('RAG_MODEL', "mixtral-instruct-v0.1")
```
All the configuration options of the context manager can be found in the [Context Manager Documentation](docs/context_manager.md) TODO.

Finally, pass the context manager object to the core classes and run the desired functions. For example, to create a new screener object with custom values, use the following code snippet:
```python
screener = Screener(context_manager=cm)
screener.do_screening()
```

```python
evaluator = Evaluator(cm)
evaluation = evaluator.get_full_evaluation()
```

```python
criteria_optimizer = CriteriaOptimizer(cm)
# There is a set of optimization methods available
optimized_criteria = criteria_optimizer.automated_feature_improvement(feature_importance) # automatic
optimized_criteria = criteria_optimizer.context_aware_feature_improvement(feature_importance) # automatic
optimized_criteria = criteria_optimizer.context_discriminative_feature_improvement(feature_importance) # automatic
optimized_criteria = criteria_optimizer.expert_feature_improvement(feature_importance, annotations(optional)) # human in the loop
optimized_criteria = criteria_optimizer.advanced_feature_improvement(feature_importance, annotations) # human in the loop


# For those that need the feature importance, it can be accessed in two ways
evaluator = Evaluator(cm)
evaluation = evaluator.get_full_evaluation()
feature_importance = evaluation[2] # either get the feature importance from the full evaluation
feature_importance = evaluator.get_feature_importance() # or get it directly
```


The core functions will take all relevant parameters from the context manager. 
The only notable exception is the criteria optimizer, which requires the feature importance from the evaluation.
> 🚨 **Please note:** 
> The results are not stored in the context manager. The results are only returned as a return value of the function.



### Graphical User Interface
_currently in to-do and not usable._
<!---
To use `aisaac` via the graphical user interface, run the following command:
   ```bash
   streamlit run uisaac.py
   ```
--->

## Testing and Benchmarking
To get started, navigate to the `testing_guides` folder and open the benchmarking notebook:
```bash
cd docs
cd testing_guides
jupyter notebook benchmark_guide.ipynb
```

## Documentation
For detailed documentation on how to use `aisaac`, refer to the [User Guide](docs/user_guide.md) TODO.

## Contributing
`aisaac` is currently closed to external contributions. However, if you have suggestions or feedback, feel free to open an issue on the repository.

## Acknowledgements
This project was developed as part of my bachelor thesis at the University of Hamburg.
Special thanks to Fernando for his guidance and support throughout the project.
Thanks to the team at the lab for their support and feedback.

## License
TODO

## Contact
TODO
