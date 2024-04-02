import json

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from aisaac.aisaac.utils import Logger


class Evaluator:
    def __init__(self, context_manager):
        self.system_manager = context_manager.get_system_manager()
        self.checkpoint_keys = list(context_manager.get_config('CHECKPOINT_DICT').keys())
        self.full_result_path = self.system_manager.get_full_path(
            f"{context_manager.get_config('RESULT_PATH')}/{context_manager.get_config('RESULT_FILE')}")
        self.full_original_result_path = self.system_manager.get_full_path(
            f"{context_manager.get_config('ORIGINAL_RESULT_PATH')}/{context_manager.get_config('ORIGINAL_RESULT_FILE')}")
        self.logger = Logger(__name__).get_logger()
        self.result_saver = context_manager.get_result_saver()

    def get_tp_tn_fp_fn(self):
        data = self.result_saver.read_csv_to_dict_relevant_only(self.full_result_path)
        gold_standard = self.result_saver.read_csv_to_dict_relevant_only(self.full_original_result_path)
        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for title in data:
            # if there is no result for this entry we want to skip it
            if data[title] is None or data[title] == "":
                continue
            relevant = (data[title] == 'True')
            if title in gold_standard:
                if gold_standard[title] == 'True':
                    if relevant:
                        tp += 1
                    else:
                        fn += 1
                else:
                    if relevant:
                        fp += 1
                    else:
                        tn += 1
        return tp, tn, fp, fn

    def calculate_mcc(self, tp, tn, fp, fn):
        # to all variables, add one so that we don't get a 0-division error
        tp += 1
        tn += 1
        fp += 1
        fn += 1
        return (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    def calculate_f_score(self, tp, fp, fn):
        return tp / (tp + 0.5 * (fp + fn))

    # Function to extract values from dictionary and return as Series
    def extract_values(self, row, key):
        dict_str = row['checkpoints']
        if pd.isna(dict_str):  # Handling NaN values
            return None
        # Preprocess the string to replace single quotes with double quotes
        dict_str = dict_str.strip().replace("'", "\"")
        dict_str = dict_str.replace('True', 'true').replace('False', 'false')

        dict_obj = json.loads(dict_str)
        return dict_obj.get(key)

    def get_results_dataframe(self):
        df_results = pd.read_csv(self.full_result_path)

        # Add columns to df for each key in the dictionary
        for key in self.checkpoint_keys:
            df_results[key] = df_results.apply(lambda row: self.extract_values(row, key), axis=1)

        # Drop the original dictionary column if needed
        df_results.drop(columns=['checkpoints'], inplace=True)
        df_results.drop(columns=['converted'], inplace=True)
        df_results.drop(columns=['embedded'], inplace=True)
        df_results.drop(columns=['reasoning'], inplace=True)
        return df_results

    def get_gold_standard_dataframe(self):
        df_results = pd.read_csv(self.full_original_result_path)
        return df_results

    def get_feature_importance(self):
        gold_standard = self.get_gold_standard_dataframe()
        predictions = self.get_results_dataframe()
        # Drop rows with missing values
        predictions_cleaned = predictions.dropna()
        # Preprocessing the data to catch any errors in the format
        # Function to convert potential string representations of boolean values to actual boolean values
        predictions_preprocessed = predictions_cleaned.replace(
            {'True': True, 'False': False, 'true': True, 'false': False})

        X = predictions_preprocessed.iloc[:, 2:].values  # Assuming features are columns 2 through last
        y = predictions_preprocessed.iloc[:, 1].values  # Assuming the second column contains the label 'relevant'

        # Initialize and train random forest classifier
        clf = RandomForestClassifier(n_estimators=500, random_state=42)
        clf.fit(X, y)

        # Get feature importances
        feature_importances = clf.feature_importances_

        # Print feature importances
        for i, importance in enumerate(feature_importances):
            self.logger.info(f"Feature {self.checkpoint_keys[i]}: Importance Score = {importance}")

        return feature_importances

    def draw_feature_importance(self, feature_importances):
        # Create a DataFrame to store feature importances with corresponding keys
        importance_df = pd.DataFrame({'Feature': self.checkpoint_keys, 'Importance': feature_importances})

        # Sort feature importances in descending order
        importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)

        plt.ylabel('Feature Importance')
        plt.bar(range(len(importance_df_sorted)), importance_df_sorted['Importance'], align='center')

        plt.xticks(range(len(importance_df_sorted)), importance_df_sorted['Feature'], rotation=90)
        plt.xlim([-1, len(importance_df_sorted)])

        plt.tight_layout()
        # plt.savefig('feature-importance.pdf', dpi=300)
        plt.show()

    def get_full_evaluation(self):
        tptnfpfn = self.get_tp_tn_fp_fn()
        mcc = self.calculate_mcc(*tptnfpfn)
        f_score = self.calculate_f_score(tptnfpfn[0], tptnfpfn[2], tptnfpfn[3])
        feature_importance = self.get_feature_importance()
        self.draw_feature_importance(feature_importance)
        return mcc, f_score, feature_importance
