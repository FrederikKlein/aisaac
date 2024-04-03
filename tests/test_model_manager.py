import unittest
from unittest.mock import patch, MagicMock

from aisaac.aisaac.utils import ModelManager


class MyTestCase(unittest.TestCase):

    @patch('ollama.list', autospec=True)  # Direct reference to ollama.list as imported
    @patch('xinference.client.Client.list_models', autospec=True)  # Patching list_models of Client directly
    @patch('aisaac.aisaac.utils.context_manager.ContextManager', autospec=True)
    def test_use_local_models_true_calls_ollama_list(self, MockContextManager, MockClient, mock_ollama_list):
        # Setup the mock to return True for LOCAL_MODELS
        mock_context_manager = MockContextManager.return_value
        mock_context_manager.get_config.side_effect = lambda key: {
            'LOCAL_MODELS': True,
            'MODEL_CLIENT_URL': 'http://localtest',
            'EMBEDDING_MODEL': 'local-embedding-model',
            'RAG_MODEL': 'local-rag-model'
        }.get(key, None)

        model_manager = ModelManager(mock_context_manager)

        # Verifying ollama.list() was called
        mock_ollama_list.assert_called_once()
        MockClient.return_value.list_models.assert_not_called()

    @patch('ollama.list', autospec=True)  # Direct reference to ollama.list as imported
    @patch('aisaac.aisaac.utils.model_manager.Client', autospec=True)  # Mocking the Client class
    @patch('aisaac.aisaac.utils.context_manager.ContextManager', autospec=True)
    def test_use_local_models_false_calls_cosy_client_list_models(self, MockContextManager, MockClient,
                                                                  mock_ollama_list):
        # Setup the mock to return False for LOCAL_MODELS
        mock_context_manager = MockContextManager.return_value
        mock_context_manager.get_config.side_effect = lambda key: {
            'LOCAL_MODELS': False,
            'MODEL_CLIENT_URL': 'http://externaltest',
            'EMBEDDING_MODEL': 'external-embedding-model',
            'RAG_MODEL': 'external-rag-model'
        }.get(key, None)

        mock_client_instance = MockClient.return_value
        mock_client_instance.list_models = MagicMock()

        mock_client_instance.authenticate = MagicMock()

        model_manager = ModelManager(mock_context_manager)

        # Verifying cosy_client.list_models() was called
        mock_client_instance.list_models.assert_called_once()
        mock_ollama_list.assert_not_called()


if __name__ == '__main__':
    unittest.main()

# %%
