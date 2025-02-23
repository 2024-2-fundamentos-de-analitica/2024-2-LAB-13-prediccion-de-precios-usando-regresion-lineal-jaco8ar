import zipfile
import pandas as pd
import numpy as np

import os
import json
import gzip
import pickle

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.feature_selection import VarianceThreshold

def read_zip_data(type_of_data):
    zip_path = f"files/input/{type_of_data}_data.csv.zip"
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        file_names = zip_file.namelist()
        with zip_file.open(file_names[0]) as file:
            file_df = pd.read_csv(file)
    return file_df

def clean_data(df):
    df = df.copy()
    df["Age"] = 2021 - df["Year"]
    df = df.drop(columns = ["Year", "Car_Name"])

    return df


def build_pipeline(categorical_features, numerical_features):
    transformer = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
    ('scaler', MinMaxScaler(), numerical_features)
    ],
    remainder='passthrough'
    )

    pipeline = Pipeline(steps = [
        ('preprocessor', transformer),
        ('variance_threshold', VarianceThreshold(threshold=0.0001)), 
        ('feature_selection', SelectKBest(score_func=f_classif, k=10)),
        ('classifier', LinearRegression())
    ])

    return pipeline

def optimize_pipeline(pipeline, X_train, y_train):
    param_grid = {
    'feature_selection__k':range(1,15),
    'classifier__fit_intercept':[True,False],
    'classifier__positive':[True,False]
    }

    grid_search = GridSearchCV(
        estimator = pipeline,
        param_grid = param_grid,
        cv = 10,
        scoring = "neg_median_absolute_error",
        n_jobs=-1, 
        
        verbose = 2
    )

    grid_search.fit(X_train, y_train)
    return grid_search, grid_search.best_estimator_

def save_model(grid_search):
    os.makedirs("files/models/", exist_ok=True)
    with gzip.open("files/models/model.pkl.gz", 'wb') as f:
        pickle.dump(grid_search, f)

def evaluate_model(model, X, y, dataset_name):

    y_pred = model.predict(X)

    metrics = {
        "type" : "metrics",
        "dataset": dataset_name,
        "r2": r2_score(y, y_pred),
        "mse": mean_squared_error(y, y_pred),
        "mad": median_absolute_error(y, y_pred)
    }
    
    return metrics

def run_job():

    objective_name = "Present_Price"

    train_data = read_zip_data("train")
    test_data = read_zip_data("test")
    


    train_data_clean = clean_data(train_data)
    test_data_clean = clean_data(test_data)

    print(train_data_clean.columns)


    X_train = train_data_clean.drop(objective_name, axis = 1)
    X_test = test_data_clean.drop(objective_name, axis = 1)

    y_train = train_data_clean[objective_name]
    y_test = test_data_clean[objective_name] 

    categorical_features = ["Fuel_Type", "Selling_type", "Transmission"]
    numerical_features= [col for col in X_train.columns if col not in categorical_features]

    pipeline  = build_pipeline(categorical_features, numerical_features)

    grid_search, best_model = optimize_pipeline(pipeline, X_train, y_train)

    save_model(grid_search)

    train_metrics = evaluate_model(best_model, X_train, y_train, "train")
    test_metrics = evaluate_model(best_model, X_test, y_test, "test")

    metrics = [train_metrics, test_metrics]

    os.makedirs("files/output/", exist_ok=True)
    with open("files/output/metrics.json", "w") as f:
        for metric in metrics:
            f.write(json.dumps(metric) + "\n")
    print("metricas y modelo guardado :)" )




if __name__ == "__main__":
    run_job()
