import pandas as pd
from scipy.stats import entropy
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

class ClusteringHyperparameters:
    def __init__(self, df_data, df_labels, random_state=42):
        self.df_values = df_data
        self.df_labels = df_labels
        self.random_state = random_state
        self.results = []

        # Map triage labels to numeric values
        triage_mapping = {
            "green": 0,
            "yellow": 1,
            "orange": 2,
            "red": 3
        }
        self.df_labels["triage_numeric"] = df_labels["triage"].map(triage_mapping)

    def calculate_entropy(self, true_labels, cluster_labels):
        contingency_matrix = pd.crosstab(true_labels, cluster_labels)
        row_sums = contingency_matrix.sum(axis=1)
        normalized_matrix = contingency_matrix.div(row_sums, axis=0)
        entropy_values = normalized_matrix.apply(lambda row: entropy(row, base=2), axis=1)
        return entropy_values.mean()

    def apply_clustering(self, cluster_model, description):
        cluster_model.fit(self.df_values.values)

        # Evaluate clustering with classic metrics
        siluetas = silhouette_score(X=self.df_values.values, labels=cluster_model.labels_)
        calinski = calinski_harabasz_score(X=self.df_values.values, labels=cluster_model.labels_)
        davies = davies_bouldin_score(X=self.df_values.values, labels=cluster_model.labels_)

        # Calculate entropy
        entropy_value = self.calculate_entropy(self.df_labels["triage_numeric"], cluster_model.labels_)
        row = [description, siluetas, calinski, davies, entropy_value]
        self.results.append(row)
        self.df_labels[description] = cluster_model.labels_

    def run_minibatch_kmeans(self, param_grid):
        grid = list(ParameterGrid(param_grid))
        for params in grid:
            try:
                description = f"minibatch_kmeans_{params}"
                cluster_model = MiniBatchKMeans(**params, random_state=self.random_state)
                print(f"Running {description}")
                self.apply_clustering(cluster_model, description)
            except Exception as e:
                print(f"Error with {description}: {e}")

    def save_results(self, performance_path="performances.csv", labels_path="labels.csv"):
        df_performances = pd.DataFrame(self.results, columns=["description", "siluetas", "calinski", "davies", "entropy"])
        df_performances.to_csv(performance_path, index=False)
        self.df_labels.to_csv(labels_path, index=False)
        print(f"\n** Results saved to {performance_path} and {labels_path} **")
