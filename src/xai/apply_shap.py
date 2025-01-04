import os
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    AdaBoostClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class SHAPAnalyzer:
    def __init__(self, model_name, data, labels, test_size=0.3, random_state=42):
        self.model_name = model_name
        self.data = data
        self.labels = labels
        self.test_size = test_size
        self.random_state = random_state
        self.train_data = None
        self.validation_data = None
        self.train_response = None
        self.validation_response = None
        self.shap_values = None
        self.explainer = None
        self.model = self._initialize_model()
    
    def _initialize_model(self):
        models = {
            "decision_tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "adaboost": AdaBoostClassifier(),
            "extra_trees": ExtraTreesClassifier(),
            "knn": KNeighborsClassifier(),
            "svc": SVC(probability=True),
        }

        return models[self.model_name]
    
    def split_data(self):
        self.train_data, self.validation_data, self.train_response, self.validation_response = train_test_split(
            self.data, self.labels, test_size=self.test_size, random_state=self.random_state
        )
    
    def fit_model(self):
        self.model.fit(self.train_data, self.train_response)
    
    def compute_shap_values(self):
        self.explainer = shap.KernelExplainer(
            model=self.model.predict,
            data=self.train_data.values,
            feature_names=self.train_data.columns.tolist()
        )
        self.shap_values = self.explainer.shap_values(self.validation_data)
    
    def plot_summary(self, plot_type="bar", max_display=10, save_path=None, title=None):
        shap.summary_plot(
            self.shap_values, self.validation_data, plot_type=plot_type, max_display=max_display, show=False
        )
        if title:
            plt.title(title)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()

    def analyze(self, save_dir="../results/xai/"):
        self.split_data()
        self.fit_model()
        self.compute_shap_values()

        os.makedirs(save_dir, exist_ok=True)

        self.plot_summary(
            plot_type="bar",
            max_display=19,
            save_path=f"{save_dir}global_importance_bar_{self.model_name}.png",
            title=f"Overall feature importance ({self.model_name})"
        )

        self.plot_summary(
            plot_type="dot",
            max_display=19,
            save_path=f"{save_dir}global_importance_scatter_{self.model_name}.png",
            title=f"Distribution of SHAP values ​​by feature ({self.model_name})"
        )
