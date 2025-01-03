import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.gridspec import GridSpec
sns.set_style("whitegrid")
plt.rc('font', size=10)

class MakeRegressionPlots(object):

    def __init__(self, dataset=None, path_export="", hue=""):
        self.palette_values = ['#026E81', '#00ABBD', '#FFB255', '#F45F74']
        self.colors = sns.color_palette(self.palette_values)

        self.dataset = dataset
        self.path_export = path_export
        self.hue = hue
        
    def plot_by_algorithm(self, name_fig="regression_performance_by_algorithm.png"):        
        df_melted = self.dataset.melt(
            id_vars=["Algorithm", "Stage"],
            value_vars=["MSE", "MAE", "R2"],
            var_name="Metric",
            value_name="Score",
        )

        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(2, 2, figure=fig)

        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0])
        ]

        metrics = ["MSE", "MAE", "R2"]

        for ax, metric in zip(axes, metrics):
            sns.barplot(
                ax=ax,
                data=df_melted[df_melted["Metric"] == metric],
                y="Algorithm",
                x="Score",
                hue="Stage",
                palette=self.colors
            )
            ax.set_title(metric)
            ax.get_legend().remove()

        # Agregar leyenda común
        handles, labels = axes[0].get_legend_handles_labels()
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
