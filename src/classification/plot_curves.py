import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from math import ceil, sqrt


class ModelEvaluation:
    def __init__(self, file_path, path_export):
        self.path_export = path_export
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

    def _create_grid(self, num_items):
        grid_size = ceil(sqrt(num_items))
        return grid_size, grid_size

    def plot_all_roc_curves(self):
        algorithms = self.results_df['algorithm'].unique()
        rows, cols = self._create_grid(len(algorithms))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)

        for ax, algorithm in zip(axes.flat, algorithms):
            filtered_df = self.results_df[self.results_df['algorithm'] == algorithm]
            if filtered_df.empty:
                continue
            data = filtered_df['roc_data'].values[0]
            for entry in data:
                ax.plot(entry['fpr'], entry['tpr'], label=f"Cluster {entry['class']}")
            ax.plot([0, 1], [0, 1], 'k--', label="Random Guess")
            ax.set_title(f"ROC Curve for {algorithm}")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.grid()

        for ax in axes.flat[len(algorithms):]: 
            ax.axis("off")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=10)
        plt.savefig(f"{self.path_export}/roc_curves.png", dpi=300)
        plt.show()

    def plot_all_pr_curves(self):
        algorithms = self.results_df['algorithm'].unique()
        rows, cols = self._create_grid(len(algorithms))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)

        for ax, algorithm in zip(axes.flat, algorithms):
            filtered_df = self.results_df[self.results_df['algorithm'] == algorithm]
            if filtered_df.empty:
                continue
            data = filtered_df['pr_data'].values[0]
            for entry in data:
                ax.plot(entry['recall'], entry['precision'], label=f"Cluster {entry['class']}")
            ax.set_title(f"Precision-Recall Curve for {algorithm}")
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.grid()

        for ax in axes.flat[len(algorithms):]: 
            ax.axis("off")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=10)
        plt.savefig(f"{self.path_export}/pr_curves.png", dpi=300)
        plt.show()

    def plot_all_confusion_matrices(self):
        algorithms = self.results_df['algorithm'].unique()
        rows, cols = self._create_grid(len(algorithms))
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), constrained_layout=True)

        for ax, algorithm in zip(axes.flat, algorithms):
            filtered_df = self.results_df[self.results_df['algorithm'] == algorithm]
            if filtered_df.empty:
                continue
            data = filtered_df['conf_matrix'].values[0]
            labels = sorted(set([item['true_label'] for item in data]))
            conf_matrix = np.zeros((len(labels), len(labels)), dtype=int)

            for entry in data:
                i = labels.index(entry['true_label'])
                j = labels.index(entry['predicted_label'])
                conf_matrix[i, j] = int(entry['count'])

            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                        xticklabels=labels, yticklabels=labels, ax=ax)
            ax.set_title(f"Confusion Matrix for {algorithm}")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")

        for ax in axes.flat[len(algorithms):]: 
            ax.axis("off")

        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.05), fontsize=10)
        plt.savefig(f"{self.path_export}/confusion_matrices.png", dpi=300)
        plt.show()
