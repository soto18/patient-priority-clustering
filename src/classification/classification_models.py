import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    matthews_corrcoef, confusion_matrix, roc_curve, auc, 
    precision_recall_curve)

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    AdaBoostClassifier, ExtraTreesClassifier)

from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize

from joblib import dump

class ClassificationModel:

    def __init__(self, 
                 train_values=None, 
                 test_values=None, 
                 train_response=None, 
                 test_response=None,
                 folder_export=None) -> None:
        
        self.train_values = train_values
        self.test_values = test_values
        self.train_response = train_response
        self.test_response = test_response
        self.folder_export = folder_export

        self.scores = ['f1_weighted', 'recall_weighted', 'precision_weighted', 'accuracy']
        self.keys = ['fit_time', 'score_time', 'test_f1_weighted', 'test_recall_weighted', 'test_precision_weighted', 'test_accuracy']

    def __get_metrics(self, y_true, y_pred):
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        precision = precision_score(y_true=y_true, y_pred=y_pred, average='weighted')
        recall = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
        mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
        sensitivity = recall
        return [accuracy, precision, recall, f1, mcc, sensitivity]

    def __process_performance_cross_val(self, performances):
        return [np.mean(performances[key]) for key in self.keys]

    def __apply_model(self, model, description):
        # Convertir a OneVsRestClassifier si es multiclase
        if len(set(self.train_response)) > 2:
            model = OneVsRestClassifier(model)

        model.fit(self.train_values, self.train_response)
        predictions = model.predict(self.test_values)

        # Binarizar etiquetas para curvas multiclase
        y_true_binarized = label_binarize(self.test_response, classes=np.unique(self.test_response))

        # Probabilidades para curvas
        if hasattr(model, "decision_function"):
            probas = model.decision_function(self.test_values)
        elif hasattr(model, "predict_proba"):
            probas = model.predict_proba(self.test_values)
        else:
            probas = None

        # Métricas de validación
        metrics_validation = self.__get_metrics(self.test_response, predictions)

        # Validación cruzada
        response_cv = cross_validate(model, self.train_values, self.train_response, cv=5, scoring=self.scores)
        metrics_cv = self.__process_performance_cross_val(response_cv)

        # Curvas ROC y precisión-recall
        fpr, tpr, roc_auc, precision, recall, ap_score = [None] * 6
        if probas is not None and y_true_binarized.shape[1] > 1:
            fpr, tpr, roc_auc, precision, recall, ap_score = [], [], [], [], [], []
            for i in range(y_true_binarized.shape[1]):
                fpr_i, tpr_i, _ = roc_curve(y_true_binarized[:, i], probas[:, i])
                precision_i, recall_i, _ = precision_recall_curve(y_true_binarized[:, i], probas[:, i])
                fpr.append(fpr_i)
                tpr.append(tpr_i)
                roc_auc.append(auc(fpr_i, tpr_i))
                precision.append(precision_i)
                recall.append(recall_i)
                ap_score.append(auc(recall_i, precision_i))
        elif probas is not None:
            fpr, tpr, _ = roc_curve(self.test_response, probas)
            precision, recall, _ = precision_recall_curve(self.test_response, probas)
            roc_auc = auc(fpr, tpr)
            ap_score = auc(recall, precision)

        # Matriz de confusión
        conf_matrix = confusion_matrix(self.test_response, predictions).tolist()

        metrics_row = [description] + metrics_cv + metrics_validation
        graph_row = [description, fpr, tpr, roc_auc, precision, recall, ap_score, conf_matrix]

        return metrics_row, graph_row

    def apply_exploring(self):
        metrics_matrix = []
        graph_data_matrix = []

        classifiers = {
            "SVC": SVC(probability=True),
            "KNN": KNeighborsClassifier(),
            "DecisionTree": DecisionTreeClassifier(),
            "RandomForest": RandomForestClassifier(),
            "AdaBoost": AdaBoostClassifier(),
            "GradientBoosting": GradientBoostingClassifier(),
            "ExtraTrees-ensemble": ExtraTreesClassifier()
        }

        for name, clf_model in classifiers.items():
            try:
                metrics_row, graph_row = self.__apply_model(clf_model, name)
                metrics_matrix.append(metrics_row)
                graph_data_matrix.append(graph_row)
                dump(clf_model, f"{self.folder_export}/{name.lower()}.joblib")
            except Exception as e:
                print(f"Error al entrenar o guardar el modelo {name}: {e}")

        metrics_header = [
            "algorithm", 'fit_time', 'score_time', 'F1_cv', 'recall_cv', 'precision_cv', 'accuracy_cv', 
            'accuracy_val', 'precision_val', 'recall_val', 'f1_val', 'matthews_corrcoef_val', 'sensitivity'
        ]

        graph_header = [
            "algorithm", 'fpr', 'tpr', 'roc_auc', 'precision', 'recall', 'ap_score', 'conf_matrix'
        ]

        df_summary = pd.DataFrame(data=metrics_matrix, columns=metrics_header)
        df_graph_data = pd.DataFrame(data=graph_data_matrix, columns=graph_header)

        return df_summary, df_graph_data