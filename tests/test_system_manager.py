import unittest
from unittest.mock import patch

from aisaac.aisaac.utils.context_manager import ContextManager
from aisaac.aisaac.utils.system_manager import SystemManager


class TestSystemManager(unittest.TestCase):

    def setUp(self):
        self.context_manager = ContextManager()
    @patch('os.makedirs')
    def test_make_directory(self, mock_makedirs):
        relative_path = 'new/directory'
        base_directory = self.context_manager.get_config('BASE_DIR')

        system_manager = SystemManager(self.context_manager)
        full_path = system_manager.make_directory(relative_path)

        mock_makedirs.assert_called_once_with(f'{base_directory}/{relative_path}', exist_ok=True)
        self.assertEqual(full_path, f'{base_directory}/{relative_path}')

    @patch('os.path.exists')
    def test_path_exists(self, mock_exists):
        base_directory = self.context_manager.get_config('BASE_DIR')
        relative_path = 'existing/directory'
        mock_exists.return_value = True  # Simulate that the directory exists

        system_manager = SystemManager(self.context_manager)
        exists = system_manager.path_exists(relative_path)

        mock_exists.assert_called_once_with(f'{base_directory}/{relative_path}')
        self.assertTrue(exists)

    @patch('shutil.rmtree')
    @patch('os.makedirs')
    def test_reset_directory(self, mock_makedirs, mock_rmtree):
        base_directory = self.context_manager.get_config('BASE_DIR')
        relative_path = 'directory/to/reset'

        system_manager = SystemManager(self.context_manager)
        system_manager.reset_directory(relative_path)

        mock_rmtree.assert_called_once_with(f'{base_directory}/{relative_path}')
        mock_makedirs.assert_called_once_with(f'{base_directory}/{relative_path}')

    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    def test_save_file(self, mock_open):
        base_directory = self.context_manager.get_config('BASE_DIR')
        relative_path = 'path/to/file.txt'
        content = 'Hello, World!'

        system_manager = SystemManager(self.context_manager)
        system_manager.save_file(relative_path, content)

        mock_open.assert_called_once_with(f'{base_directory}/{relative_path}', 'wb')
        mock_open().write.assert_called_once_with(content)

    @patch('os.path.join')
    def test_get_full_path(self, mock_join):
        base_directory = self.context_manager.get_config('BASE_DIR')
        relative_path = 'path/to/resource'
        expected_full_path = '/base/directory/path/to/resource'
        mock_join.return_value = expected_full_path

        system_manager = SystemManager(self.context_manager)
        full_path = system_manager.get_full_path(relative_path)

        mock_join.assert_called_once_with(base_directory, relative_path)
        self.assertEqual(full_path, expected_full_path)

    @patch('os.remove')
    @patch('os.path.isfile')
    def test_delete_file(self, mock_isfile, mock_remove):
        base_directory = self.context_manager.get_config('BASE_DIR')
        relative_path = 'path/to/delete.txt'
        mock_isfile.return_value = True  # Simulate that the file exists

        system_manager = SystemManager(self.context_manager)
        system_manager.delete_file(relative_path)

        mock_remove.assert_called_once_with(f'{base_directory}/{relative_path}')


if __name__ == '__main__':
    unittest.main()
