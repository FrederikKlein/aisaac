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
`aisaac` can be used as a python package or through its graphical user interface (GUI).

### Python Package
To use `aisaac` as a python package, import the `aisaac` module and use its classes and functions as needed. 
The core classes are
- 'Screener'
- 'Evaluator'
- 'Optimizer'

To organize contexts, the core classes use a context manager that is filled with default values. The context manager can be used to set custom values for the classes.
To do so, first import the context manager:
```python
from aisaac.aisaac.utils.context_manager import ContextManager
```
Then, create a context manager object and set custom values:
```python
cm = ContextManager()
cm.set_config('RAG_MODEL', "mixtral:latest")
cm.set_config('RESULT_FILE', "results.csv")
cm.set_config('CHECKPOINT_DICTIONARY', checkpoint_dict)
```
Finally, pass the context manager object to the core classes and run the desired functions. For example, to create a new screener object with custom values, use the following code snippet:
```python
screener = Screener(context_manager=cm)
screener.do_screening()
```
All the configuration options of the context manager can be found in the [Context Manager Documentation](docs/context_manager.md) TODO.



### Graphical User Interface
To use `aisaac` via the graphical user interface, run the following command:
   ```bash
   streamlit run uisaac.py
   ```

## Testing and Benchmarking
To ensure aisaac meets various user needs and integrates smoothly with a wide range of configurations, especially concerning different language models (LLMs), we've developed a customizable testing script. This script facilitates the creation and execution of tests across numerous configurations, enabling both developers and users to verify aisaac's compatibility and performance under different conditions. We highly encourage users and contributors to leverage this script to expand our testing coverage, particularly for new or custom LLMs. Detailed instructions on how to use the script and contribute tests can be found in the tests directory. Your contributions help make aisaac more robust and versatile, ensuring it remains effective across diverse deployment scenarios.
Find our interactive testing notebook in the `testing_guides` directory. This resource is designed to help you understand `aisaac`'s testing framework and to create your tests. It combines executable code with comprehensive instructions, making it an invaluable tool for ensuring `aisaac` meets your specific requirements.

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
