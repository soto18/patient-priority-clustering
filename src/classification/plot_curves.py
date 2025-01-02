import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class ModelEvaluation:
    def __init__(self, file_path):
        self.results_df = pd.read_csv(file_path)
        self.results_df['roc_data'] = self.results_df['roc_data'].apply(self.safe_eval)
        self.results_df['pr_data'] = self.results_df['pr_data'].apply(self.safe_eval)
        self.results_df['conf_matrix'] = self.results_df['conf_matrix'].apply(self.safe_eval)

    @staticmethod
    def safe_eval(data):
        if isinstance(data, str):
            data = data.replace('np.int64', '')
            try:
                return ast.literal_eval(data)
            except (ValueError, SyntaxError):
                pass
        return None

    def plot_roc_curve(self, algorithm):
        filtered_df = self.results_df[self.results_df['algorithm'] == algorithm]
        if filtered_df.empty:
            raise ValueError(f"No data found for algorithm '{algorithm}'.")
        data = filtered_df['roc_data'].values[0]
        plt.figure(figsize=(8, 6))
        for entry in data:
            plt.plot(entry['fpr'], entry['tpr'], label=f"Cluster {entry['class']}")
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
        plt.title(f"ROC Curve for {algorithm}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_pr_curve(self, algorithm):
        filtered_df = self.results_df[self.results_df['algorithm'] == algorithm]
        if filtered_df.empty:
            raise ValueError(f"No data found for algorithm '{algorithm}'.")
        data = filtered_df['pr_data'].values[0]
        plt.figure(figsize=(8, 6))
        for entry in data:
            plt.plot(entry['recall'], entry['precision'], label=f"Cluster {entry['class']}")
        plt.title(f"Precision-Recall Curve for {algorithm}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_confusion_matrix(self, algorithm):
        filtered_df = self.results_df[self.results_df['algorithm'] == algorithm]
        data = filtered_df['conf_matrix'].values[0]
        labels = sorted(set([item['true_label'] for item in data]))
        conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)

        for entry in data:
            i = labels.index(entry['true_label'])
            j = labels.index(entry['predicted_label'])
            conf_matrix[i, j] = int(entry['count'])

        plt.figure(figsize=(4, 3))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
        plt.title(f"Confusion Matrix for {algorithm}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
