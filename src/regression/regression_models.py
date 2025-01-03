import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, mean_absolute_percentage_error
)
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
from joblib import dump

warnings.filterwarnings("ignore")

class RegressionModel:
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

        self.scores = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
        self.keys = ['fit_time', 'score_time', 'test_neg_mean_squared_error', 
                     'test_neg_mean_absolute_error', 'test_r2']
        os.makedirs(self.folder_export, exist_ok=True)

    def __get_metrics(self, y_true, y_pred):
        mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
        r2 = r2_score(y_true=y_true, y_pred=y_pred)
        evs = explained_variance_score(y_true=y_true, y_pred=y_pred)
        mape = mean_absolute_percentage_error(y_true=y_true, y_pred=y_pred)
        return [mse, mae, r2, evs, mape]

    def __process_performance_cross_val(self, performances):
        return [np.mean(performances[key]) for key in self.keys]

    def __apply_model(self, model, description):
        model.fit(self.train_values, self.train_response)
        predictions = model.predict(self.test_values)

        # Guardar predicciones
        predictions_df = pd.DataFrame({
            "real_values": self.test_response,
            "predicted_values": predictions
        })
        predictions_df.to_csv(f"{self.folder_export}/{description.lower()}_predictions.csv", index=False)

        # Métricas de validación
        metrics_validation = self.__get_metrics(self.test_response, predictions)

        # Validación cruzada
        response_cv = cross_validate(model, self.train_values, self.train_response, cv=5, scoring=self.scores)
        metrics_cv = self.__process_performance_cross_val(response_cv)

        # Almacenar métricas
        metrics_row = [description] + metrics_cv + metrics_validation

        # Guardar el modelo entrenado
        dump(model, f"{self.folder_export}/{description.lower()}.joblib")

        return metrics_row

    def apply_regression(self):
        metrics_matrix = []

        # Modelos a evaluar
        models = [
            (LinearRegression(), "LinearRegression"),
            (ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42), "ElasticNet"),
            (DecisionTreeRegressor(random_state=42), "DecisionTree"),
            (RandomForestRegressor(random_state=42, n_estimators=100), "RandomForest"),
            (GradientBoostingRegressor(random_state=42, n_estimators=100), "GradientBoosting"),
            (SVR(kernel='rbf', C=1.0, epsilon=0.1), "SVR"),
        ]

        for model, description in models:
            try:
                metrics_row = self.__apply_model(model, description)
                metrics_matrix.append(metrics_row)
            except Exception as e:
                print(f"Error in model {description}: {e}")

        # Encabezado para el resumen de métricas
        metrics_header = [
            "algorithm", 'fit_time', 'score_time', 'MSE_cv', 'MAE_cv', 'R2_cv', 
            'MSE_val', 'MAE_val', 'R2_val', 'ExplainedVariance', 'MAPE'
        ]

        df_summary = pd.DataFrame(data=metrics_matrix, columns=metrics_header)
        df_summary.to_csv(f"{self.folder_export}/results_exploration.csv", index=False)

        return df_summary
