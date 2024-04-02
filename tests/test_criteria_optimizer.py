import unittest
from unittest.mock import patch, MagicMock

from aisaac.aisaac.core.criteria_optimizer import CriteriaOptimizer  # Adjust import according to your project structure


class TestCriteriaOptimizer(unittest.TestCase):

    def setUp(self):
        self.mock_context_manager = MagicMock()
        self.mock_context_manager.get_logger.return_value = MagicMock()
        self.mock_context_manager.get_result_saver.return_value = MagicMock()
        self.mock_context_manager.get_model_manager.return_value = MagicMock()
        self.mock_context_manager.get_document_manager.return_value = MagicMock()
        self.mock_context_manager.get_similarity_searcher.return_value = MagicMock()
        self.mock_context_manager.get_config.side_effect = lambda key: {
            'FEATURE_IMPORTANCE_THRESHOLD': '0.5',
            'MAX_FEATURE_IMPROVEMENT_DOCUMENTS': '10',
            'IMPORTANCE_GREATER_THAN_THRESHOLD': True,
            'NUMBER_EXPERT_CHOICES': '3'
        }.get(key, None)

        self.optimizer = CriteriaOptimizer(self.mock_context_manager)

    @patch.object(CriteriaOptimizer, '_CriteriaOptimizer__generate_new_checkpoints', return_value="New Checkpoint")
    def test_automated_feature_improvement(self, mock_generate_new_checkpoints):
        checkpoint_dictionary = {"cp1": "Initial"}
        checkpoint_importances = [0.6]  # Assume this triggers modification

        improved_checkpoints = self.optimizer.automated_feature_improvement(checkpoint_dictionary, checkpoint_importances)

        mock_generate_new_checkpoints.assert_called()


if __name__ == '__main__':
    unittest.main()
