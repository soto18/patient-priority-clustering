from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class ViewGroups:
    def __init__(self, df_values=None, df_labels=None):
        self.df_values = df_values
        self.df_labels = df_labels

    def visualize_best_partition(self, cluster_column, save_path):
        # PCA
        pca = PCA(n_components=2)
        reduced_data_pca = pca.fit_transform(self.df_values.values)
        
        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data_tsne = tsne.fit_transform(self.df_values.values)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        unique_clusters = self.df_labels[cluster_column].unique()
        num_clusters = len(unique_clusters)
        colormap = cm.get_cmap('viridis', num_clusters)
        cluster_colors = {cluster: colormap(i) for i, cluster in enumerate(unique_clusters)}

        # Visualización con PCA
        axes[0].set_title(f"Clusters visualized using PCA")
        for cluster in unique_clusters:
            cluster_data_pca = reduced_data_pca[self.df_labels[cluster_column] == cluster]
            axes[0].scatter(cluster_data_pca[:, 0], cluster_data_pca[:, 1], 
                            label=f"Cluster {cluster}", 
                            alpha=0.7, 
                            color=cluster_colors[cluster])
        
        axes[0].set_xlabel("p0")
        axes[0].set_ylabel("p1")
        axes[0].legend(title="Clusters")
        axes[0].grid(alpha=0.2)

        # Visualización con t-SNE
        axes[1].set_title(f"Clusters visualized using t-SNE")
        for cluster in unique_clusters:
            cluster_data_tsne = reduced_data_tsne[self.df_labels[cluster_column] == cluster]
            axes[1].scatter(cluster_data_tsne[:, 0], cluster_data_tsne[:, 1], 
                            label=f"Cluster {cluster}", 
                            alpha=0.7, 
                            color=cluster_colors[cluster])
        
        axes[1].set_xlabel("t0")
        axes[1].set_ylabel("t1")
        axes[1].legend(title="Clusters")
        axes[1].grid(alpha=0.2)
        
        plt.tight_layout()
        plt.savefig(save_path, format='png', bbox_inches='tight')
        plt.show()

    def visualize_clusters_pca(self, algorithms, save_path):
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.df_values.values)
        
        n_algorithms = len(algorithms)
        n_cols = 3 
        n_rows = (n_algorithms + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        # Generar un gráfico por cada algoritmo
        for i, algorithm in enumerate(algorithms):
            ax = axes[i]
            num_clusters = len(set(self.df_labels[algorithm]))
            scatter = ax.scatter(
                reduced_data[:, 0], reduced_data[:, 1], 
                c=self.df_labels[algorithm], cmap='viridis', alpha=0.7
            )
            
            ax.set_title(f"{algorithm} - Clusters: {num_clusters}")
            ax.set_xlabel("p0")
            ax.set_ylabel("p1")
            ax.grid(alpha=0.2)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Cluster")
        
        for j in range(len(algorithms), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(save_path, format='png', bbox_inches='tight')
        plt.show()



    def visualize_clusters_tsne(self, algorithms, save_path):
        tsne = TSNE(n_components=2, random_state=42)
        reduced_data = tsne.fit_transform(self.df_values.values)
        
        n_algorithms = len(algorithms)
        n_cols = 3 
        n_rows = (n_algorithms + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        axes = axes.flatten()
        
        for i, algorithm in enumerate(algorithms):
            ax = axes[i]
            num_clusters = len(set(self.df_labels[algorithm]))
            scatter = ax.scatter(
                reduced_data[:, 0], reduced_data[:, 1], 
                c=self.df_labels[algorithm], cmap='viridis', alpha=0.7
            )
            
            ax.set_title(f"{algorithm} - Clusters: {num_clusters}")
            ax.set_xlabel("t0")
            ax.set_ylabel("t1")
            ax.grid(alpha=0.2)
            
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Cluster")
        
        for j in range(len(algorithms), len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()
        plt.savefig(save_path, format='png', bbox_inches='tight')
        plt.show()