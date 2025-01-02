import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.gridspec import GridSpec
sns.set_style("whitegrid")
plt.rc('font', size=10)

class MakePlots(object):

    def __init__(
            self, 
            dataset=None, 
            path_export="",
            hue=""):
        
        self.palette_values = ['#026E81', '#00ABBD', '#FFB255', '#F45F74']
        self.colors = sns.color_palette(self.palette_values)

        self.dataset = dataset
        self.path_export = path_export
        self.hue = hue
        
    def plot_by_algorithm(self, name_fig="ml_classic_performance_by_algorithm.png"):        
        df_melted = self.dataset.melt(
            id_vars=["Algorithm", self.hue],
            value_vars=["Accuracy", "Precision", "Recall", "F1"],
            var_name="Metric",
            value_name="Score",
        )

        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, 2, figure=fig)

        ax_data1 = fig.add_subplot(gs[0, 0])
        ax_data2 = fig.add_subplot(gs[0, 1])
        ax_data3 = fig.add_subplot(gs[1, 0])
        ax_data4 = fig.add_subplot(gs[1, 1])

        # Gráficos de barras agrupadas
        barplot1 = sns.barplot(
            ax=ax_data1,
            data=df_melted[df_melted["Metric"] == "Accuracy"],
            y="Algorithm",
            x="Score",
            hue=self.hue,
            palette=self.colors
        )
        ax_data1.set_title("Accuracy")
        ax_data1.get_legend().remove()

        sns.barplot(
            ax=ax_data2,
            data=df_melted[df_melted["Metric"] == "Precision"],
            y="Algorithm",
            x="Score",
            hue=self.hue,
            palette=self.colors
        )
        ax_data2.set_title("Precision")
        ax_data2.get_legend().remove()

        sns.barplot(
            ax=ax_data3,
            data=df_melted[df_melted["Metric"] == "Recall"],
            y="Algorithm",
            x="Score",
            hue=self.hue,
            palette=self.colors
        )
        ax_data3.set_title("Recall")
        ax_data3.get_legend().remove()

        sns.barplot(
            ax=ax_data4,
            data=df_melted[df_melted["Metric"] == "F1"],
            y="Algorithm",
            x="Score",
            hue=self.hue,
            palette=self.colors
        )
        ax_data4.set_title("F1")
        ax_data4.get_legend().remove()

        handles, labels = barplot1.get_legend_handles_labels()
        fig.legend(
            handles, 
            labels, 
            loc='lower center', 
            bbox_to_anchor=(0.5, -0.05), 
            ncol=len(labels), 
            frameon=False
        )

        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.savefig(f"{self.path_export}/{name_fig}", dpi=300)
