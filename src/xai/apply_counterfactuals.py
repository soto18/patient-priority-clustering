import dice_ml
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from dice_ml import Model, Dice

warnings.filterwarnings('ignore')

class CounterfactualExplainer:
    def __init__(self, data_path, model=DecisionTreeClassifier(), test_size=0.3, random_state=42):
        self.df_data = pd.read_csv(data_path)
        self.data_values = self.df_data.drop(columns=['label'])
        self.responses = self.df_data['label']

        self.train_data, self.validation_data, self.train_response, self.validation_response = train_test_split(
            self.data_values, self.responses, random_state=random_state, test_size=test_size
        )

        self.continuous_features = [
            "blood pressure", "bmi", "cholesterol", "insulin", "max heart rate",
            "plasma glucose", "skin_thickness", "age"
        ]

        self.categorical_features = [
            "gender", "exercise angina", "hypertension", "heart_disease", 
            "Residence_type_Rural", "Residence_type_Urban", "smoking_status_Unknown", 
            "smoking_status_formerly smoked", "smoking_status_never smoked", 
            "smoking_status_smokes", "chest pain type"
        ]

        self.outcome_name = "label"
        self.data_dice = dice_ml.Data(
            dataframe=self.df_data,
            continuous_features=self.continuous_features,
            categorical_features=self.categorical_features,
            outcome_name=self.outcome_name
        )

        self.model = model
        self.model.fit(self.train_data, self.train_response)
        self.model_dice = Model(model=self.model, backend="sklearn")
        self.dice = Dice(self.data_dice, self.model_dice)

    def generate_and_show_counterfactuals(self, query_instance, total_cfs=5):
        counterfactuals_list = []
        for i in range(len(self.model.classes_)): # modelo multiclase
            counterfactuals = self.dice.generate_counterfactuals(
                query_instances=query_instance,
                total_CFs=total_cfs,
                desired_class=i
            )
            counterfactuals_df = counterfactuals.visualize_as_dataframe()
            counterfactuals_list.append(counterfactuals_df)
        
        return counterfactuals_list
