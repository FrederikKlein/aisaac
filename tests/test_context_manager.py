import unittest

from aisaac.aisaac.utils.context_manager import ContextManager


class TestContextManager(unittest.TestCase):
    def test_get_config(self):
        # Test getting a configuration variable
        self.assertEqual(ContextManager.get_config('CHROMA_PATH'), "chroma")

    def test_set_config(self):
        # Test setting a configuration variable
        ContextManager.set_config('CHROMA_PATH', '/path/to/new/chroma')
        self.assertEqual(ContextManager.get_config('CHROMA_PATH'), '/path/to/new/chroma')

    def test_get_nonexistent_config(self):
        # Test getting a nonexistent configuration variable
        self.assertIsNone(ContextManager.get_config('NONEXISTENT_KEY'))


if __name__ == '__main__':
    unittest.main()
