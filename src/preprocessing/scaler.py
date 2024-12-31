import pandas as pd
from sklearn.preprocessing import RobustScaler
from joblib import dump

class DataScaler:
    def __init__(self, data):
        self.df_data = data

    def separating_data(self):
        binary_cols = self.df_data.columns[self.df_data.nunique() == 2]
        binary_cols = binary_cols.append(pd.Index(['chest pain type']))
        non_binary_cols = self.df_data.columns.difference(binary_cols)
        df_binary = self.df_data[binary_cols]
        df_non_binary = self.df_data[non_binary_cols]
        return df_binary, df_non_binary

    def scale_data(self, df_non_binary):
        self.age = df_non_binary["age"].tolist()
        df_non_binary = df_non_binary.drop("age", axis=1)
        scaler_instance = RobustScaler()
        scaler_instance.fit(df_non_binary.values)
        data_scaled = scaler_instance.transform(df_non_binary.values)
        df_scaled = pd.DataFrame(data=data_scaled, columns=df_non_binary.columns)
        return df_scaled, scaler_instance

    def save_results(self, df_binary, df_scaled, scaler_instance, output_csv, output_scaler):
        df_concat = pd.concat([df_binary, df_scaled], axis=1)
        age_series = pd.Series(self.age, name='age')
        df_scaled = pd.concat([df_concat, age_series], axis=1)
        df_scaled.to_csv(output_csv, index=False)
        dump(scaler_instance, output_scaler)