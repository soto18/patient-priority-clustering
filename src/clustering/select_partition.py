import warnings
warnings.filterwarnings('ignore')

class BestPartition:
    def __init__(self, df_performance=None):
        self.df_performance = df_performance

    def select_best_partition(df_performance):
        # Filtrar las particiones con Silhouette mayor a 0.35
        filtered_df = df_performance[df_performance["siluetas"] > 0.35]
            
        # Calcular un score ponderado
        filtered_df["score"] = (
            filtered_df["siluetas"] +   # Mayor Silhouette
            filtered_df["calinski"] -   # Mayor Calinski-Harabasz
            filtered_df["davies"] -     # Menor Davies
            filtered_df["entropy"]      # Baja entropia
        )
        
        # Seleccionar la mejor partición según el score
        best_partition = filtered_df.sort_values("score", ascending=False).iloc[0]
        return filtered_df, best_partition