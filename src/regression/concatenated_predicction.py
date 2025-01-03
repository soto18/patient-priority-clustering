import os
import pandas as pd
import math
import matplotlib.pyplot as plt

class ViewPredictions:
    def __init__(self, folder_export=None):
        self.folder_export = folder_export

    def concatenate_predictions_by_algorithm(self):
        predictions_dict = {}

        for file in os.listdir(self.folder_export):
            if file.endswith("_predictions.csv"):
                model_name = file.replace("_predictions.csv", "")
                df = pd.read_csv(os.path.join(self.folder_export, file))
                predictions_dict["real_values"] = df["real_values"]                
                predictions_dict[model_name] = df["predicted_values"]

        concatenated_df = pd.DataFrame(predictions_dict)
        concatenated_df.to_csv(f"{self.folder_export}/all_predictions_values.csv", index=False)

        return concatenated_df


    def plot_predictions_values(self, concatenated_df=None):
        names = {
            "decisiontree": "Decision Tree",
            "gradientboosting": "Gradient Boosting",
            "svr": "SVM Regressor",
            "linearregression": "Linear Regression",
            "elasticnet": "Elastic Net",
            "randomforest": "Random Forest"
        }
        concatenated_df = concatenated_df.rename(columns=names) 

        real_values = concatenated_df["real_values"]
        algorithms = concatenated_df.columns[1:]

        num_algorithms = len(algorithms)
        cols = 3 
        rows = math.ceil(num_algorithms / cols)

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4), squeeze=False)
        axes = axes.flatten()  # Aplanar la matriz de ejes

        for i, algo in enumerate(algorithms):
            axes[i].scatter(range(len(real_values)), real_values, label="Real values", color="black", alpha=0.6, s=10)
            axes[i].scatter(range(len(real_values)), concatenated_df[algo], label="Predicted values", color="red", alpha=0.6, s=10)

            axes[i].set_title(f"{algo}", fontsize=10)
            axes[i].set_xlabel("Example", fontsize=8)
            axes[i].set_ylabel("Values", fontsize=8)
            axes[i].grid(True, linestyle='--', alpha=0.7)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=10)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(f"{self.folder_export}/predicction_values.png", dpi=300)
        plt.show()
