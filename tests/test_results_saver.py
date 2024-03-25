import unittest
from unittest.mock import MagicMock, mock_open, patch

from aisaac.aisaac.utils.result_saver import ResultSaver  # Adjust this import according to your project structure


class TestResultSaver(unittest.TestCase):

    def setUp(self):
        # Mock context manager and its return values
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.get_config.side_effect = lambda key: {
            'RESULT_PATH': '/fake/result/path',
            'RESULT_FILE': 'results.csv',
            'CHROMA_PATH': '/fake/chroma/path',
            'RESET_RESULTS': 'True'
        }[key]

        # Patch 'open' here, before instantiating ResultSaver
        patcher = patch('builtins.open', mock_open())
        self.addCleanup(patcher.stop)  # Ensure patch is cleaned up after tests
        self.mock_file = patcher.start()

        # Now instantiate ResultSaver
        self.result_saver = ResultSaver(self.mock_context_manager)

    @patch('os.path.join', return_value='/fake/result/path/results.csv')
    @patch('builtins.open', new_callable=mock_open)
    def test_write_csv(self, mock_file, mock_join):
        self.result_saver.write_csv([{'title': 'Test', 'converted': True}])
        mock_file.assert_called_once_with('/fake/result/path/results.csv', 'w', newline='')
        mock_file().write.assert_called()  # Check if write was called, can be more specific if needed

    @patch('builtins.open', new_callable=mock_open, read_data='title,converted\nTest,True\n')
    def test_read_csv_to_dict_list(self, mock_file):
        result_list = self.result_saver.read_csv_to_dict_list()
        self.assertEqual(result_list, [{'title': 'Test', 'converted': 'True'}])
        mock_file.assert_called_once_with('/fake/result/path/results.csv', 'r')

    @patch('aisaac.aisaac.utils.result_saver.ResultSaver.read_csv_to_dict_list', return_value=[{'title': 'Test', 'converted': True}])
    @patch('aisaac.aisaac.utils.result_saver.ResultSaver.write_csv')
    def test_update_csv(self, mock_write_csv, mock_read_csv):
        self.result_saver.update_csv([{'title': 'Test', 'converted': False}])
        mock_read_csv.assert_called_once()
        mock_write_csv.assert_called_once()

    @patch('aisaac.aisaac.utils.result_saver.ResultSaver.read_csv_to_dict_list', return_value=[{'title': 'Old', 'converted': True}])
    @patch('aisaac.aisaac.utils.result_saver.ResultSaver.write_csv')
    def test_add_data_csv(self, mock_write_csv, mock_read_csv):
        self.result_saver.add_data_csv([{'title': 'New', 'converted': False}])
        mock_read_csv.assert_called_once()
        mock_write_csv.assert_called_once_with(
            [{'title': 'Old', 'converted': True}, {'title': 'New', 'converted': False}])

    @patch('builtins.open', new_callable=mock_open)
    def test_reset_results(self, mock_file):
        self.result_saver.reset_results()
        mock_file.assert_called_once_with('/fake/result/path/results.csv', 'w', newline='')

    # Additional tests for create_new_result_entry, get_result_list, and set_up_new_results_file can be added similarly


if __name__ == '__main__':
    unittest.main()
