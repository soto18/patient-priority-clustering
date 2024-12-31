import seaborn as sns
import matplotlib.pyplot as plt

class CorrelationAnalysis:
    def __init__(self, data, custom_palette=None):
        self.df_data = data
        self.custom_palette = custom_palette or ["#F1948A", "#85C1E9"]
        self.colors = sns.color_palette(self.custom_palette)

    def plot_heatmap(self, columns, figsize=(19, 10)):
        correlation_matrix = self.df_data[columns].corr()

        plt.figure(figsize=figsize)
        sns.heatmap(
            correlation_matrix,
            cmap=sns.light_palette(self.custom_palette[0], as_cmap=True),
            annot=True,
            fmt=".2f",
            cbar_kws={"shrink": .8}
        )
        plt.xticks(rotation=90, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()