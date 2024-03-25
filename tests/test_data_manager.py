import unittest

from aisaac.aisaac.utils.context_manager import ContextManager
from aisaac.aisaac.utils.data_manager import DocumentManager, VectorDataManager


class TestDocumentDataLoader(unittest.TestCase):
    def setUp(self):
        self.skipTest("Document structure not implemented yet.")

        # Set up a mock ContextManager instance with dummy configuration
        self.context_manager = ContextManager()
        self.context_manager.set_config('DATA_PATHS', ['Data/Excluded', 'Data/Included'])
        self.context_manager.set_config('DATA_FORMAT', '*.pdf')
        self.context_manager.set_config('RANDOM_SUBSET', 'True')
        self.context_manager.set_config('SUBSET_SIZE', '5')
        self.context_manager.set_config('CHROMA_PATH', 'chroma')


    def test_load_data(self):
        # Initialize DataManager with the mock ContextManager
        data_manager = DocumentManager(self.context_manager)
        # Ensure that the DataManager loads data correctly
        data = data_manager.load_data('Data/Excluded')
        self.assertIsNotNone(data)
        self.assertTrue(isinstance(data, list))

    def test_get_data(self):
        # Initialize DataManager with the mock ContextManager
        data_manager = DocumentManager(self.context_manager)
        # Ensure that the DataManager returns data correctly
        data = data_manager.get_data()
        self.assertIsNotNone(data)
        self.assertTrue(isinstance(data, list))

    def test_get_runnable_data(self):
        # Initialize DataManager with the mock ContextManager
        data_manager = DocumentManager(self.context_manager)
        # Ensure that the DataManager returns runnable data correctly
        runnable_data = data_manager.get_runnable_data()
        self.assertIsNotNone(runnable_data)
        self.assertTrue(isinstance(runnable_data, list))


from unittest.mock import MagicMock, patch


class TestVectorDataManager(unittest.TestCase):
    @patch('vdm.Logger')  # Mock the Logger to avoid actual logging
    @patch('vdm.ContextManager')  # Mock the ContextManager
    def setUp(self, MockContextManager, MockLogger):
        self.skipTest("Document structure not implemented yet.")

        # Setup mock context manager and its return values
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.get_config.side_effect = lambda key: \
            {'APPLY_SENTENCE_SPLITTING_CHUNKING': 'False', 'CHUNK_SIZE': '100', 'CHUNK_OVERLAP': '20'}[key]
        self.mock_context_manager.get_model_manager.return_value = MagicMock()
        self.mock_context_manager.get_document_data_manager.return_value = MagicMock()
        self.mock_context_manager.get_system_manager.return_value = MagicMock()
        self.mock_context_manager.get_results_saver.return_value = MagicMock()

        # Initialize the VDM class instance for testing
        self.vdm = VectorDataManager(self.mock_context_manager)


    @patch('vdm.NLTKTextSplitter')
    def test_chunk_documents_sentence_splitting(self, MockTextSplitter):
        # Set the configuration to apply sentence splitting
        self.vdm.apply_sentence_splitting_chunking = True

        mock_splitter = MockTextSplitter.return_value
        mock_splitter.split_text.return_value = ['chunk1', 'chunk2']

        documents = ['doc1', 'doc2']
        chunks = self.vdm.chunk_documents(documents)

        self.assertEqual(len(chunks), 2)
        self.assertListEqual(chunks, ['chunk1', 'chunk2'])
        MockTextSplitter.assert_called_once()
        self.vdm.logger.info.assert_called()

    @patch('vdm.RecursiveCharacterTextSplitter')
    def test_chunk_documents_character_splitting(self, MockTextSplitter):
        self.vdm.apply_sentence_splitting_chunking = False

        mock_splitter = MockTextSplitter.return_value
        mock_splitter.split_documents.return_value = ['chunk1', 'chunk2']

        documents = ['doc1']
        chunks = self.vdm.chunk_documents(documents)

        self.assertEqual(len(chunks), 2)
        self.assertListEqual(chunks, ['chunk1', 'chunk2'])
        MockTextSplitter.assert_called_once_with(chunk_size=100, chunk_overlap=20, length_function=len,
                                                 add_start_index=True)
        self.vdm.logger.info.assert_called()

    @patch('vdm.Chroma')
    def test_save_to_chroma(self, MockChroma):
        chunks = ['chunk1', 'chunk2']
        title = 'test_title'
        path = 'test_path'

        self.vdm.save_to_chroma(chunks, title, path)
        self.vdm.system_manager.make_directory.assert_called_once_with(path)
        MockChroma.from_documents.assert_called_once()
        self.vdm.logger.info.assert_called()
        self.vdm.logger.debug.assert_called()

        # Test error handling
        MockChroma.from_documents.side_effect = Exception('Test error')
        self.vdm.save_to_chroma(chunks, title, path)
        self.vdm.logger.error.assert_called_with(f"Error saving chunks to {path}: Test error")

    # Add more tests for other methods like create_document_stores, etc.


if __name__ == '__main__':
    unittest.main()

# %%
