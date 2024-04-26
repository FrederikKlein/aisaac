#%% md
# Here's a quickstart guide to get you up and running with the package.
#%% md
# The code is currently divided into the functionalities it provides. The main functionalities are:
# 1. Screening a set of PDFs or MDs against inclusion/ecxlusion criteria.
# 2. Evaluating the quality of the screening process against a gold standard.
# 3. Optimizing the criteria.
# 
# Q: _But wouldn't it be easier to just have one god class that does everything?_ 
# A: Sure, but once you understand the logic of the architecture, you'll see that it's quite straightforward. No code gets loaded or executed until it's absolutely necessary and the seperation of concerns is intact. 
# I get that every coder follows different practices, so I will define a class in this notebook though that you can easily copy if you'd rather have everything in one place.
#%% md
# First, let's talk
# # Imports
#%%
# depending which of the functionalities you want to use, you can import the following classes
from aisaac.aisaac.core.screener import Screener
from aisaac.aisaac.core.criteria_optimizer import CriteriaOptimizer
from aisaac.aisaac.core.evaluator import Evaluator

# if you want to change any of the default settings (which will probably be necessary), you can import the context_manager class
from aisaac.aisaac.utils.context_manager import ContextManager

# these two imports will come in handy
import time
import csv
#%% md
# # Changing the settings
# Now, let's have a look at the context manager. This class is responsible for managing the configuration and facilitating the integration of various utility modules such as data handling, model management, and result saving within an application context. Although at first glance this might seem more complex, it really isn't. As long as you use the context manager, there will be no clashes, no dataloss or dataleaks due to wrong parameters.
# You do not have to change all of the settings, only those that are relevant to you and your use case. If you want to have a more in depth look of the context manager, have a look at the context manager documentation in the docs folder.
#%%
# Creating a ContextManager instance with custom configurations
custom_config = {
    'DATA_PATHS': ["Data/MyData"],
    'CHECKPOINT_DICTIONARY': {"Checkpoint 1": "this is the checkpoint", "Checkpoint 2": "this is another checkpoint"}
}
context_manager = ContextManager(config=custom_config)
#%%
# you can change configurations afterwards but be aware that these might lead to conflicts
context_manager.set_config('DATA_PATHS', ["Data/NewData"])
#%%
# you can also get the current configuration
context_manager.get_config('DATA_PATHS')
#%% md
# The relevant directory configurations are the following:
#%%
custom_config = {
    'BASE_PATH': 'BaseDir',
    'BIN_PATH': "bin",
    'CHROMA_PATH': "chroma",
    'DATA_PATHS': ["Data/MyData"],
    'RESULT_PATH': "results",
    'ORIGINAL_RESULT_PATH': "gold_standard_data",
    'RESULT_FILE': "results.csv", # will be created automatically
    'ORIGINAL_RESULT_FILE': "gold_standard.csv", # which is in the ORIGINAL_RESULT_PATH
}
#%% md
# # Screening
# 
# Let's talk about the screening process. I assume that there is a directory structure that looks like this:
# 
# ```
# BaseDir
# │
# └───bin
# │
# └───chroma
# │
# └───Data
# │   │ MyData
# │   │   │ file1.pdf
# │   │   │ file2.pdf
# │   │   │ ...
# │
# └───gold_standard_data
# │   │ gold_standard.csv (doesn't have to exist yet)
# │
# └───results
# │   │ results.csv (doesn't have to exist yet, will be created automatically)
# ```
# 
# Drop all the files you want to screen in the MyData folder. Set the checkpoints in the context manager. Set which models you want to use in the screener. Update the data_manager to the new data. If not happened yet, create the embeddings.
# Then you can start the screening process.
# 
# Currently, the creation of the directories does not happen automatically. Also updating the data_manager once you uploaded all your data does not happen automatically. This is to prevent unnecessary data loading and saving, but can be automated in the future.
#%%
initial_cm = ContextManager()
# set the relevant configurations
initial_cm.set_config('BASE_DIR', "BaseDir")
initial_cm.set_config('BIN_PATH', "bin")
initial_cm.set_config('CHROMA_PATH', "chroma")
initial_cm.set_config('DATA_PATHS', ["Data/MyData"])
initial_cm.set_config('RESULT_PATH', "results")
initial_cm.set_config('ORIGINAL_RESULT_PATH', "gold_standard_data")
initial_cm.set_config('RESULT_FILE', "results.csv")
# choose your models and whether you want to use local models
initial_cm.set_config('LOCAL_MODELS', "False")
initial_cm.set_config('MODEL_CLIENT_URL', "https://llm.cosy.bio")
initial_cm.set_config('EMBEDDING_MODEL', "gte-large")
initial_cm.set_config('RAG_MODEL', "llama-2-chat")

# these commands are currently still necessary to do manually, they will be automated in the future
initial_cm.get_document_data_manager().update_global_data() # once all the data is uploaded
initial_cm.get_vector_data_manager().create_document_stores() # after updating the global data, creates the embeddings
#%%
screener = Screener(initial_cm)
screener.do_screening()
#%% md
# The results will be found in the result file you specified.
#%%
