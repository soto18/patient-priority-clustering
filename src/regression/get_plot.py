import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, "../")
import pandas as pd
from src.regression.plot_performance import MakeRegressionPlots


class PerformanceMergerAndPlotter:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df_data = None
        self.df_processed = None

    def load_data(self, file_name):
        self.df_data = pd.read_csv(f"{self.input_path}/{file_name}")
        print(f"Data loaded from {self.input_path}/{file_name}")

    def merge_documents(self):
        if self.df_data is None:
            raise ValueError("No data loaded. Use 'load_data' first.")

        df_results_train = self.df_data[['algorithm', 'MSE_cv', 'MAE_cv', 'R2_cv']]
        df_results_train.columns = ["Algorithm", 'MSE', 'MAE', 'R2']

        df_results_test = self.df_data[['algorithm', 'MSE_val', 'MAE_val', 'R2_val']]
        df_results_test.columns = ["Algorithm", 'MSE', 'MAE', 'R2']

        df_results_train["Stage"] = "Training"
        df_results_test["Stage"] = "Validation"

        df_process = pd.concat([df_results_train, df_results_test], axis=0)
        df_process.reset_index(inplace=True)

        self.df_processed = df_process

    def save_processed_data(self, file_name):
        if self.df_processed is None:
            raise ValueError("No processed data to save. Use 'merge_documents' first.")

        self.df_processed.to_csv(f"{self.output_path}/{file_name}", index=False)
        print(f"Processed data saved to {self.output_path}/{file_name}")

    def plot_data(self):
        if self.df_processed is None:
            raise ValueError("No processed data available for plotting. Use 'merge_documents' first.")

        print("Plotting data...")
        make_plots = MakeRegressionPlots(dataset=self.df_processed, path_export=self.output_path, hue="Stage")
        make_plots.plot_by_algorithm()
        print("Plots generated.")

    def run(self, input_file, output_file):

        self.load_data(input_file)
        self.merge_documents()
        self.save_processed_data(output_file)
        self.plot_data()
