import unittest

from aisaac.aisaac.utils.context_manager import ContextManager


class TestContextManager(unittest.TestCase):
    def setUp(self):
        # Set up the ContextManager instance
        self.context_manager = ContextManager()
    def test_get_config(self):
        # Test getting a configuration variable
        self.assertEqual(self.context_manager.get_config('CHROMA_PATH'), "chroma")

    def test_set_config(self):
        # Test setting a configuration variable
        self.context_manager.set_config('CHROMA_PATH', '/path/to/new/chroma')
        self.assertEqual(self.context_manager.get_config('CHROMA_PATH'), '/path/to/new/chroma')

    def test_get_nonexistent_config(self):
        # Test getting a nonexistent configuration variable
        self.assertIsNone(self.context_manager.get_config('NONEXISTENT_KEY'))


if __name__ == '__main__':
    unittest.main()
