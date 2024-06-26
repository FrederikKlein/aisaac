import json
import math
import unicodedata

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from aisaac.aisaac.utils import Logger


class Evaluator:
    def __init__(self, context_manager):
        self.system_manager = context_manager.get_system_manager()
        self.checkpoint_keys = list(context_manager.get_config('CHECKPOINT_DICTIONARY').keys())
        self.full_result_path = self.system_manager.get_full_path(
            f"{context_manager.get_config('RESULT_PATH')}/{context_manager.get_config('RESULT_FILE')}")
        self.full_original_result_path = self.system_manager.get_full_path(
            f"{context_manager.get_config('ORIGINAL_RESULT_PATH')}/{context_manager.get_config('ORIGINAL_RESULT_FILE')}")
        self.logger = Logger(__name__).get_logger()
        self.result_saver = context_manager.get_result_saver()

    def get_tp_tn_fp_fn(self):
        y_true, y_pred = self.get_y_true_y_pred()

        tp = sum((y_true == True) & (y_pred == True))
        tn = sum((y_true == False) & (y_pred == False))
        fp = sum((y_true == False) & (y_pred == True))
        fn = sum((y_true == True) & (y_pred == False))

        return tp, tn, fp, fn

    def get_completion_rate(self):
        data = self.result_saver.read_csv_to_dict_relevant_only(self.full_result_path)
        total = len(data)
        completed = 0
        for title in data:
            # check if there is a result for this entry
            if title == '.idea':
                total -= 1
                continue
            if data[title] is not None and data[title] != "":
                completed += 1
        return completed / total

    def calculate_mcc(self, tp, tn, fp, fn):
        # to all variables, add one so that we don't get a 0-division error
        tp += 1
        tn += 1
        fp += 1
        fn += 1
        return (tp * tn - fp * fn) / ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5

    def calculate_f_score(self, tp, fp, fn):
        return tp / (tp + 0.5 * (fp + fn))

    def calculate_accuracy(self, tp, tn, fp, fn):
        return (tp + tn) / (tp + tn + fp + fn)

    def calculate_specificity(self, tn, fp):
        return tn / (tn + fp)

    def calculate_sensitivity(self, tp, fn):
        return tp / (tp + fn)

    def calculate_fowlkes_mallows_index(self, tp, tn, fp, fn):
        tp += 1
        tn += 1
        fp += 1
        fn += 1
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return math.sqrt(precision * recall)

    def generate_confusion_matrix(self, tp, tn, fp, fn):
        return {
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn
        }

    def draw_confusion_matrix(self, confusion_matrix):
        plt.figure(figsize=(10, 7))
        plt.imshow([[confusion_matrix['TP'], confusion_matrix['FN']], [confusion_matrix['FP'], confusion_matrix['TN']]],
                   cmap='Blues')
        plt.colorbar()
        plt.xticks([0, 1], ['Predicted Positive', 'Predicted Negative'])
        plt.yticks([0, 1], ['Actual Positive', 'Actual Negative'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()

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

    def get_results_dataframe_title_relevant_column(self):
        df_results = self.get_results_dataframe()
        df_results = df_results.loc[:, ['title', 'relevant']]
        return df_results

    def get_gold_standard_dataframe(self):
        df_results = pd.read_csv(self.full_original_result_path)
        return df_results

    def get_feature_importance(self):
        clf = self.get_trained_classifier()

        # Get feature importances
        feature_importances = clf.feature_importances_

        # Print feature importances
        for i, importance in enumerate(feature_importances):
            self.logger.info(f"Feature {self.checkpoint_keys[i]}: Importance Score = {importance}")

        return feature_importances

    def get_trained_classifier(self):
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
        return clf

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

    def get_y_true_y_pred(self):
        gold_standard = self.get_gold_standard_dataframe()
        predictions = self.get_results_dataframe_title_relevant_column()
        # Drop rows with missing values
        gold_standard_cleaned = gold_standard.dropna()
        predictions_cleaned = predictions.dropna()

        # Remove leading and trailing quotation marks and normalize umlauts in the gold standard
        gold_standard_cleaned.iloc[:, 0] = gold_standard_cleaned.iloc[:, 0].str.strip('"').apply(normalize_string)

        # Normalize umlauts in the predictions cleaned
        predictions_cleaned.iloc[:, 0] = predictions_cleaned.iloc[:, 0].apply(normalize_string)
        # Ensure alignment by merging on a common identifier (assuming the first column is the identifier)
        merged = gold_standard_cleaned.merge(predictions_cleaned, on=gold_standard_cleaned.columns[0],
                                             suffixes=('_gold', '_pred'))

        # Drop rows with any missing values in the merged dataframe
        merged_cleaned = merged.dropna()
        merge_preprocessed = merged_cleaned.replace({'True': True, 'False': False, 'true': True, 'false': False})
        # merge_preprocessed = merged_cleaned

        # Extract y_true and y_pred from the cleaned, merged dataframe
        y_true = merge_preprocessed.iloc[:, 1].values
        y_pred = merge_preprocessed.iloc[:, 2].values
        return y_true, y_pred

    def get_cohen_kappa(self):
        from sklearn.metrics import cohen_kappa_score
        y_true, y_pred = self.get_y_true_y_pred()
        return cohen_kappa_score(y_true, y_pred)

    def calculate_pabak(self):
        y_true, y_pred = self.get_y_true_y_pred()
        # Ensure y_true and y_pred are not empty
        if len(y_true) > 0 and len(y_pred) > 0:
            # Calculate the observed agreement
            agreement_array = np.array(y_true) == np.array(y_pred)
            # Ensure there is at least one True value
            if np.any(agreement_array):
                agreement = np.mean(agreement_array)
                # Calculate PABAK
                pabak = (2 * agreement) - 1
                return pabak
        return np.nan  # Return NaN if conditions are not met

    def get_benchmarking_scores(self):
        tptnfpfn = self.get_tp_tn_fp_fn()
        mcc = self.calculate_mcc(*tptnfpfn)
        kappa = self.get_cohen_kappa()
        accuracy = self.calculate_accuracy(*tptnfpfn)
        fmi = self.calculate_fowlkes_mallows_index(*tptnfpfn)
        f_score = self.calculate_f_score(tptnfpfn[0], tptnfpfn[2], tptnfpfn[3])
        specificity = self.calculate_specificity(tptnfpfn[1], tptnfpfn[2])
        sensitivity = self.calculate_sensitivity(tptnfpfn[0], tptnfpfn[3])
        completion_rate = self.get_completion_rate()
        return mcc, kappa, accuracy, fmi, f_score, specificity, sensitivity, completion_rate

    def get_full_evaluation(self):
        tptnfpfn = self.get_tp_tn_fp_fn()
        mcc = self.calculate_mcc(*tptnfpfn)
        f_score = self.calculate_f_score(tptnfpfn[0], tptnfpfn[2], tptnfpfn[3])
        feature_importance = self.get_feature_importance()
        self.draw_feature_importance(feature_importance)
        confusion_matrix = self.generate_confusion_matrix(*tptnfpfn)
        self.draw_confusion_matrix(confusion_matrix)
        specificity = self.calculate_specificity(tptnfpfn[1], tptnfpfn[2])
        sensitivity = self.calculate_sensitivity(tptnfpfn[0], tptnfpfn[3])
        cohens_kappa = self.get_cohen_kappa()
        fmi = self.calculate_fowlkes_mallows_index(*tptnfpfn)
        completion_rate = self.get_completion_rate()
        return mcc, f_score, feature_importance, confusion_matrix, specificity, sensitivity, cohens_kappa, fmi, completion_rate

    def get_evaluation_dict(self):
        y_true, y_pred = self.get_y_true_y_pred()
        record2answer = self.result_saver.read_csv_to_dict_relevant_only(self.full_result_path)
        # drop all the records that are not in the original result file
        missing_records = self.result_saver.read_csv_to_dict_relevant_only(self.full_original_result_path)

        return {
            "TP": self.get_tp_tn_fp_fn()[0],
            "TN": self.get_tp_tn_fp_fn()[1],
            "FP": self.get_tp_tn_fp_fn()[2],
            "FN": self.get_tp_tn_fp_fn()[3],
            "Confusion Matri x": self.generate_confusion_matrix(*self.get_tp_tn_fp_fn()),
            "ratio_of_completion": self.get_completion_rate(),
            "Precision": self.calculate_precision(),
            "Recall": self.calculate_recall(),
            "F1-score": self.calculate_f1_score(),
            "Matthews correlation coefficient": self.calculate_mcc(*self.get_tp_tn_fp_fn()),
            "Cohen's kappa": self.get_cohen_kappa(),
            "PABAK": self.calculate_pabak(),
            'ratio_of_completion': self.get_completion_rate(),
            'succesfully_analyzed_articles': sum(self.get_tp_tn_fp_fn()),
            'articles_that_did_not_have_predictions': 159 - sum(self.get_tp_tn_fp_fn()),
        }

    def calculate_precision(self):
        tp, fp = self.get_tp_tn_fp_fn()[0], self.get_tp_tn_fp_fn()[2]
        return tp / (tp + fp)

    def calculate_recall(self):
        tp, fn = self.get_tp_tn_fp_fn()[0], self.get_tp_tn_fp_fn()[3]
        return tp / (tp + fn)

    def calculate_f1_score(self):
        precision = self.calculate_precision()
        recall = self.calculate_recall()
        return 2 * precision * recall / (precision + recall)


def normalize_string(s):
    # Normalize the string to NFC form
    return unicodedata.normalize('NFC', s)

# %%
