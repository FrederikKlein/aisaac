# Contents of tests/test_logger.py
import logging
import unittest
from io import StringIO

from aisaac.aisaac.utils.logger import Logger  # Import the Logger class from the logger module


class TestLogger(unittest.TestCase):

    def setUp(self):
        # Redirect logs to a stream to capture them for verification
        self.log_stream = StringIO()
        handler = logging.StreamHandler(self.log_stream)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s: %(message)s"))

        self.logger = Logger("test_logger").get_logger()
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)

    def test_debug_log(self):
        test_message = "This is a debug test"
        self.logger.debug(test_message)
        self.assertIn(test_message, self.log_stream.getvalue())

    def test_info_log(self):
        test_message = "This is an info test"
        self.logger.info(test_message)
        self.assertIn(test_message, self.log_stream.getvalue())

    def test_warning_log(self):
        test_message = "This is a warning test"
        self.logger.warning(test_message)
        self.assertIn(test_message, self.log_stream.getvalue())

    def test_error_log(self):
        test_message = "This is an error test"
        self.logger.error(test_message)
        self.assertIn(test_message, self.log_stream.getvalue())

    def test_critical_log(self):
        test_message = "This is a critical test"
        self.logger.critical(test_message)
        self.assertIn(test_message, self.log_stream.getvalue())

    def tearDown(self):
        # Clean up by removing the attached handler
        self.logger.handlers = []


if __name__ == '__main__':
    unittest.main()

#%%
