import unittest
from unittest.mock import patch, MagicMock

from aisaac.aisaac.core.screener import Screener


class TestScreener(unittest.TestCase):
    def test_init(self):
        mock_context_manager = MagicMock()
        screener = Screener(mock_context_manager)
        self.assertIsNotNone(screener.result_saver)
        self.assertIsNotNone(screener.mm)
        self.assertIsNotNone(screener.dm)
        self.assertIsNotNone(screener.similarity_searcher)

    @patch('aisaac.aisaac.utils.Logger')
    def test_do_screening(self, mock_logger):
        mock_context_manager = MagicMock()
        screener = Screener(mock_context_manager)
        screener.dm.get_runnable_titles.return_value = ['Title1', 'Title2']

        with patch.object(screener, 'craft_screening_response_for') as mock_craft:
            screener.do_screening({'checkpoint1': 'Check1'})

            # Check if craft_screening_response_for was called for each title
            self.assertEqual(mock_craft.call_count, 2)
            mock_craft.assert_any_call('Title1', {'checkpoint1': 'Check1'})
            mock_craft.assert_any_call('Title2', {'checkpoint1': 'Check1'})

    @patch('aisaac.aisaac.core.screener.StructuredOutputParser.parse')
    @patch('aisaac.aisaac.utils.Logger')
    def test_craft_screening_response_for(self, mock_logger, mock_parse):
        mock_context_manager = MagicMock()
        mock_parse.return_value = {'expected': 'parsed_output'}
        screener = Screener(mock_context_manager)
        checkpoints = {'checkpoint1': 'Check1'}
        title = 'Title1'
        expected_prompt = 'Some prompt'

        # Mock dependencies
        screener.create_context_text = MagicMock(return_value='Some context')
        screener.create_prompt = MagicMock(return_value=expected_prompt)
        screener.mm.get_rag_model.return_value.predict = MagicMock(return_value='Some response')

        # Execute
        response = screener.craft_screening_response_for(title, checkpoints)

        # Assertions
        screener.create_prompt.assert_called_once_with('Some context', checkpoints,
                                                       screener.get_output_parser().get_format_instructions())
        screener.mm.get_rag_model().predict.assert_called_once_with(expected_prompt)
        self.assertEqual(response, {'expected': 'parsed_output'})


if __name__ == '__main__':
    unittest.main()
