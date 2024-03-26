import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from aisaac.aisaac.utils.data_manager import DocumentManager, \
    VectorDataManager  # Adjust this import according to your project structure


class TestDocumentManager(unittest.TestCase):

    def setUp(self):
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.get_config.side_effect = lambda key: {
            'DATA_PATHS': ['/fake/data/path'],
            'DATA_FORMAT': '*.txt',
            'RANDOM_SUBSET': 'False',
            'SUBSET_SIZE': '10',
            'CHROMA_PATH': '/fake/chroma/path'
        }[key]
        self.document_manager = DocumentManager(self.mock_context_manager)

    @patch('aisaac.aisaac.utils.data_manager.DirectoryLoader')
    def test_load_data(self, MockDirectoryLoader):
        mock_loader = MockDirectoryLoader.return_value
        mock_loader.load.return_value = ['doc1', 'doc2', 'doc3']
        documents = self.document_manager._DocumentManager__load_data('/fake/data/path')
        self.assertEqual(documents, ['doc1', 'doc2', 'doc3'])
        MockDirectoryLoader.assert_called_once_with('/fake/data/path', glob='*.txt')

    # Additional tests within TestDocumentManager class

    @patch('aisaac.aisaac.utils.data_manager.DocumentManager._DocumentManager__load_data')
    def test_load_global_data(self, mock_load_data):
        mock_load_data.return_value = ['doc1', 'doc2']
        self.document_manager.load_global_data()
        mock_load_data.assert_called_with('/fake/data/path')
        self.assertEqual(self.document_manager.global_data, [['doc1', 'doc2']])

    @patch('random.sample')
    def test_get_data_random_subset(self, mock_random_sample):
        self.document_manager.global_data = [['doc1', 'doc2', 'doc3']]
        self.document_manager.random_subset = True
        mock_random_sample.return_value = ['doc1']
        data = self.document_manager.get_data()
        mock_random_sample.assert_called_once_with(['doc1', 'doc2', 'doc3'], 10)
        self.assertIn('doc1', data)

    def test_get_data_no_subset(self):
        self.document_manager.global_data = [['doc1', 'doc2', 'doc3']]
        self.document_manager.random_subset = False
        data = self.document_manager.get_data()
        self.assertEqual(data, ['doc1', 'doc2', 'doc3'])

    def test_get_runnable_data(self):
        # Create mock documents that mimic the structure expected by your method
        mock_doc1 = MagicMock(spec=Document)
        mock_doc1.metadata = {'source': '/path/to/doc1.pdf'}
        mock_doc2 = MagicMock(spec=Document)
        mock_doc2.metadata = {'source': '/path/to/doc2.pdf'}
        self.document_manager.global_data = [[mock_doc1, mock_doc2]]

        with patch('os.path.exists', return_value=False):
            data = self.document_manager.get_runnable_data()
            self.assertEqual(data, [])  # No documents should be returned since the paths don't exist

        # Now test with existing paths
        with patch('os.path.exists', return_value=True):
            data = self.document_manager.get_runnable_data()
            self.assertEqual(len(data), 2)  # Both documents should be returned since the paths exist
            self.assertEqual(data, [mock_doc1, mock_doc2])


class TestVectorDataManager(unittest.TestCase):

    def setUp(self):
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.get_config.side_effect = lambda key: {
            'APPLY_SENTENCE_SPLITTING_CHUNKING': 'True',
            'CHUNK_SIZE': '100',
            'CHUNK_OVERLAP': '20',
            'CHROMA_PATH': '/fake/chroma/path'
        }[key]
        self.vector_data_manager = VectorDataManager(self.mock_context_manager)

    @patch('aisaac.aisaac.utils.data_manager.NLTKTextSplitter')
    def test_chunk_documents_sentence_splitting(self, MockTextSplitter):
        mock_splitter = MockTextSplitter.return_value
        mock_splitter.split_text.return_value = ['chunk1', 'chunk2']
        documents = ['doc1', 'doc2']
        chunks = self.vector_data_manager.chunk_documents(documents)
        self.assertEqual(len(chunks), 2)
        self.assertListEqual(chunks, ['chunk1', 'chunk2'])
        MockTextSplitter.assert_called_once()

    # Additional tests within TestVectorDataManager class

    @patch('aisaac.aisaac.utils.data_manager.RecursiveCharacterTextSplitter')
    def test_chunk_documents_recursive_splitting(self, MockTextSplitter):
        self.vector_data_manager.apply_sentence_splitting_chunking = False
        mock_splitter = MockTextSplitter.return_value
        mock_splitter.split_documents.return_value = ['chunk1', 'chunk2']
        documents = ['doc1']
        chunks = self.vector_data_manager.chunk_documents(documents)
        self.assertEqual(len(chunks), 2)
        MockTextSplitter.assert_called_once_with(
            chunk_size=100, chunk_overlap=20, length_function=len, add_start_index=True
        )

    @patch('aisaac.aisaac.utils.data_manager.Chroma')
    def test_save_to_chroma(self, MockChroma):
        chunks = [MagicMock(spec=Document)]
        title = 'test_title'
        path = '/fake/chroma/path/test_title'
        self.vector_data_manager.save_to_chroma(chunks, title, path)
        MockChroma.from_documents.assert_called_once()
        self.vector_data_manager.system_manager.make_directory.assert_called_with(path)

    @patch('aisaac.aisaac.utils.data_manager.VectorDataManager.save_to_chroma')
    @patch('aisaac.aisaac.utils.data_manager.VectorDataManager.chunk_documents')
    def test_create_document_stores(self, mock_chunk_documents, mock_save_to_chroma):
        mock_context_manager = MagicMock()
        vector_data_manager = VectorDataManager(mock_context_manager)

        mock_context_manager.get_system_manager.return_value.path_exists.side_effect = [False,
                                                                                        True]  # First document is new, second already exists
        mock_chunk_documents.return_value = ['chunk1', 'chunk2']  # Mocked chunk data

        mock_context_manager.get_document_data_manager.return_value.get_data.return_value = [
            MagicMock(metadata={'source': '/fake/path/doc1.txt'}),
            MagicMock(metadata={'source': '/fake/path/doc2.txt'})
        ]
        # Execute the method under test
        vector_data_manager.create_document_stores()

        # Assertions for system and result management
        vector_data_manager.system_manager.reset_directory.assert_called_once_with(vector_data_manager.full_chroma_path)
        vector_data_manager.result_manager.reset_results.assert_called_once()

        # Assertions for document processing
        self.assertEqual(mock_chunk_documents.call_count, 1)  # Only called once for the new document
        mock_save_to_chroma.assert_called_once_with(['chunk1', 'chunk2'], 'doc1',
                                                    f"{vector_data_manager.full_chroma_path}/doc1")
        vector_data_manager.result_manager.update_result_list.assert_called_with('doc1', 'embedded', True)

    @patch('aisaac.aisaac.utils.data_manager.VectorDataManager.get_vectorstore')
    def test_get_vectorstores(self, mock_get_vectorstore):
        mock_get_vectorstore.return_value = MagicMock()  # Mock a Chroma instance
        self.vector_data_manager.document_data_manager.get_all_titles.return_value = ['title1', 'title2']
        vectorstores = self.vector_data_manager.get_vectorstores()
        self.assertEqual(len(vectorstores), 2)
        mock_get_vectorstore.assert_any_call('title1')
        mock_get_vectorstore.assert_any_call('title2')


# Additional tests for get_unified_vectorstore, get_vectorstore_with_sigmoid_relevance_score_fn can be added similarly


if __name__ == '__main__':
    unittest.main()

# %%
