import os
import unittest
from unittest.mock import patch, mock_open

from aisaac.aisaac.utils.results_saver import ResultSaver


class TestResultSaver(unittest.TestCase):

    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('shutil.rmtree')
    def test_reset_results(self, mock_rmtree, mock_exists, mock_makedirs):
        # Setup
        result_path = 'test_results'
        result_file = 'test.csv'
        chroma_path = 'test_chroma'

        # Test when chroma_path exists
        mock_exists.return_value = True
        rs = ResultSaver(result_path, result_file, reset_results=True, chroma_path=chroma_path)
        rs.reset_results()

        mock_rmtree.assert_called_once_with(chroma_path)
        mock_makedirs.assert_called_once_with(chroma_path)

        mock_rmtree.reset_mock()
        mock_makedirs.reset_mock()

        # Test when chroma_path does not exist
        mock_exists.return_value = False
        rs.reset_results()

        mock_rmtree.assert_not_called()
        mock_makedirs.assert_called_with(chroma_path)

    @patch('csv.DictWriter')
    @patch('builtins.open', new_callable=mock_open)
    def test_write_csv(self, mock_file, mock_dict_writer):
        result_path = 'test_results'
        result_file = 'test.csv'
        rs = ResultSaver(result_path, result_file)

        test_data = [{'title': 'test1', 'converted': True}]
        rs.write_csv(test_data)

        mock_file.assert_called_once_with(os.path.join(result_path, result_file), 'w', newline='')
        mock_dict_writer.return_value.writeheader.assert_called_once()
        mock_dict_writer.return_value.writerows.assert_called_once_with(test_data)

    @patch('builtins.open', new_callable=mock_open, read_data='title,converted\nTest,True\n')
    def test_read_csv_to_dict_list(self, mock_file):
        result_path = 'test_results'
        result_file = 'test.csv'
        rs = ResultSaver(result_path, result_file)

        expected_data = [{'title': 'Test', 'converted': 'True'}]
        actual_data = rs.read_csv_to_dict_list()

        self.assertEqual(actual_data, expected_data)
        mock_file.assert_called_once_with(os.path.join(result_path, result_file), 'r')

    # Add more tests for other methods like add_data_csv, get_result_list, etc.


if __name__ == '__main__':
    unittest.main()

#%%
