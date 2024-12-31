import warnings
import pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import IsolationForest


class OutlierDetection:
    def __init__(self, df_data):
        self.df_data = df_data
        self.columns_numeric = [
            'age', 'blood pressure', 'cholesterol', 'max heart rate', 
            'plasma glucose', 'skin_thickness', 'insulin', 'bmi'
        ]

    def detect_outliers_iqr(self):
        outliers_list = []
        df_numeric = self.df_data[self.columns_numeric]

        for column in df_numeric:
            Q1 = df_numeric[column].quantile(0.25)
            Q3 = df_numeric[column].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = self.df_data[(df_numeric[column] < lower_bound) | (df_numeric[column] > upper_bound)][column]

            for outlier in outliers:
                outliers_list.append({'Column': column, 'Outlier Value': outlier})

        outliers_df = pd.DataFrame(outliers_list)
        return outliers_df

    def filter_isolation_forest(self):
        df_numeric = self.df_data[self.columns_numeric]
        df_categorical = self.df_data[self.df_data.columns.difference(self.columns_numeric)]

        model = IsolationForest(contamination=0.1, random_state=42)
        model.fit(df_numeric)

        # Hacer predicción de outliers (-1 para outliers, 1 para normales)
        df_numeric['outlier'] = model.predict(df_numeric)

        df_concat = pd.concat([df_numeric, df_categorical], axis=1)
        return df_concat