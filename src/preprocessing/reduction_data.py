from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

class DimensionalityReducer:
    def __init__(self, palette=None):

        self.palette = palette if palette else sns.color_palette("deep")
        self.df_pca = None
        self.df_tsne = None

    def reduce_dimensions(self, df_values, n_components=2, perplexity=30):
        # PCA
        pca_instance = PCA(n_components=n_components, random_state=42)
        pca_transform = pca_instance.fit_transform(df_values.values)
        self.df_pca = pd.DataFrame(data=pca_transform, columns=["p0", "p1"])
        
        # t-SNE
        tsne_instance = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
        tsne_transform = tsne_instance.fit_transform(df_values.values)
        self.df_tsne = pd.DataFrame(data=tsne_transform, columns=["p0", "p1"])

    def generate_plots(self, df_labels, hue_col, save_path=None):
        self.df_pca[hue_col] = df_labels[hue_col]
        self.df_tsne[hue_col] = df_labels[hue_col]
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # PCA Plot
        sns.scatterplot(
            ax=axes[0],
            data=self.df_pca,
            x="p0", y="p1", hue=hue_col,
            palette=self.palette
        )
        axes[0].set_title("PCA")
        axes[0].legend(loc="best")
        
        # t-SNE Plot
        sns.scatterplot(
            ax=axes[1],
            data=self.df_tsne,
            x="p0", y="p1", hue=hue_col,
            palette=self.palette
        )
        axes[1].set_title("t-SNE")
        axes[1].legend(loc="best")
                
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/pca_tsne_panel.png")
        plt.show()
