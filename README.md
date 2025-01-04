# Patient priority analysis using Machine Learning and Explainable AI

This repository contains the source files and supplementary information for the development and evaluation of machine learning models to analyze patient prioritization using clustering and classification strategies. The project integrates preprocessing techniques, feature scaling, clustering, classification, regression, and Explainable AI (XAI).

---

## Índice

- [Summary of proposed work](#summary)
- [Requirements and setup](#requirements)
- [Raw Data and preprocessing](#data)
- [Clustering and cluster visualization](#clustering)
- [Classification and regression models](#models)
- [Explainable AI (XAI)](#xai)
- [Implemented Pipeline](#pipeline)

---

<a name="summary"></a>

## Summary of the Proposed Work

The goal of this project is to analyze patient prioritization using advanced machine learning techniques and Explainable AI. Through clustering and classification methods, models are developed to categorize patient priorities, complemented by techniques that facilitate the interpretation of results.

The pipeline includes:  
- Data preprocessing and handling of outliers.  
- Dimensionality reduction techniques such as PCA and t-SNE.  
- Clustering using unsupervised methods and hyperparameter variation.  
- Classification and regression models for outcome prediction.  
- Explainable AI methods like SHAP and counterfactual analysis to interpret model decisions.  

The models were validated using evaluation metrics such as precision, recall, and ROC curves.  

---

<a name="requirements"></a>

## Requirements and Installation

The requirements are summarized in the `environment.yml` file. Key dependencies include:

- Python version 3.9+  
- scikit-learn  
- pandas  
- matplotlib  
- seaborn  
- SHAP  
- t-SNE  

To install the environment:

```bash
conda env create -f environment.yml
conda activate patient-priority

```

---

<a name="data"></a>

## Raw data and preprocessing

- The raw data is available in the `input` folder as `patient_priority.csv`.  
- Data preprocessing is implemented in the notebook [`01_dataset_preprocessing.ipynb`](notebook/01_dataset_preprocessing.ipynb).  
- Key steps:
  - Outlier evaluation: [`03_evaluation_outliers.ipynb`](notebook/03_evaluation_outliers.ipynb).  
  - Correlation analysis: [`04_correlation_analysis.ipynb`](notebook/04_correlation_analysis.ipynb).  
  - Data scaling: [`05_scaler_data.ipynb`](notebook/05_scaler_data.ipynb).  
  - Data visualization: [`06_reduction_data.ipynb`](notebook/06_reduction_data.ipynb).  

Processed datasets are saved in the `results/preprocessing` folder.

---

<a name="clustering"></a>

## Clustering and cluster visualization with dimensionality reduction

Clustering methods were explored to group patients based on similarities:  
- **Algorithms Applied:**  
  - K-Means (`n_clusters=4`)  
  - MiniBatch K-Means (`n_clusters=4`)  
  - Agglomerative Clustering (`n_clusters=4`)  
  - Birch (`n_clusters=4`)  
  - Spectral Clustering (`n_clusters=4`)  
  - DBSCAN  
  - OPTICS  
  - MeanShift  
  - Affinity Propagation  

- **Dimensionality Reduction Techniques:**  
  - PCA  
  - t-SNE  

Key notebooks:  
- Clustering application: [`07_apply_clustering.ipynb`](notebook/07_apply_clustering.ipynb).  
- Clustering analysis: [`08_clustering_analysis.ipynb`](notebook/08_clustering_analysis.ipynb).  

Visualizations are available in the `results/clustering` folder.

---

<a name="models"></a>

## Classification and Regression Models

The project implements multiclass classification and regression models for the "blood pressure" descriptor:  
- **Classification Algorithms**: SVC, KNN, DecisionTree, RandomForest, AdaBoost, GradientBoosting, and ExtraTrees  
- **Regression Algorithms**: LinearRegression, ElasticNet, DecisionTree, RandomForestRegressor, GradientBoostingRegressor, and SVR  

### Notebooks

- **Classification Model Training**: [`10_apply_classification_model.ipynb`](notebook/10_apply_classification_model.ipynb)  
- **Classification Model Analysis**: [`11_classification_analysis.ipynb`](notebook/11_classification_analysis.ipynb)  
- **Regression Model Training**: [`12_apply_regression_model.ipynb`](notebook/12_apply_regression_model.ipynb)  
- **Regression Model Analysis**: [`13_regression_analysis.ipynb`](notebook/13_regression_analysis.ipynb)  

### Results

Results, including confusion matrices and ROC curves, are available in `results/classification` and `results/regression`  





