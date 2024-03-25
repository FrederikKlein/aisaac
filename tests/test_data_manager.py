import unittest

from aisaac.aisaac.utils.context_manager import ContextManager
from aisaac.aisaac.utils.data_manager import DataManager


class TestDataLoader(unittest.TestCase):
    def setUp(self):
        # Set up a mock ContextManager instance with dummy configuration
        self.context_manager = ContextManager()
        self.context_manager.set_config('DATA_PATHS', ['Data/Excluded', 'Data/Included'])
        self.context_manager.set_config('DATA_FORMAT', '*.pdf')
        self.context_manager.set_config('RANDOM_SUBSET', 'True')
        self.context_manager.set_config('SUBSET_SIZE', '5')
        self.context_manager.set_config('CHROMA_PATH', 'chroma')

    def test_load_data(self):
        # Initialize DataManager with the mock ContextManager
        data_manager = DataManager(self.context_manager)
        # Ensure that the DataManager loads data correctly
        data = data_manager.load_data('Data/Excluded')
        self.assertIsNotNone(data)
        self.assertTrue(isinstance(data, list))

    def test_get_data(self):
        # Initialize DataManager with the mock ContextManager
        data_manager = DataManager(self.context_manager)
        # Ensure that the DataManager returns data correctly
        data = data_manager.get_data()
        self.assertIsNotNone(data)
        self.assertTrue(isinstance(data, list))

    def test_get_runnable_data(self):
        # Initialize DataManager with the mock ContextManager
        data_manager = DataManager(self.context_manager)
        # Ensure that the DataManager returns runnable data correctly
        runnable_data = data_manager.get_runnable_data()
        self.assertIsNotNone(runnable_data)
        self.assertTrue(isinstance(runnable_data, list))


if __name__ == '__main__':
    unittest.main()
