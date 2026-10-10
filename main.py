import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = 'model.pkl'
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_att, cat_att):
    # Numerical pipeline
    num_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy='median')),
        ("standardize", StandardScaler())
    ])
    
    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    
    # ColumnTransformer
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_att),
        ("cat", cat_pipeline, cat_att)
    ])
    
    return full_pipeline
    
if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("housing.csv")
    housing['income_cat'] = pd.cut(housing['median_income'], bins=[0,1.5,3,4.5,6.0,np.inf], labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    for train, test in split.split(housing, housing['income_cat']):
        housing_test = housing.loc[test].drop('income_cat',axis=1).to_csv("input.csv")
        housing = housing.loc[train].drop('income_cat',axis=1)

    housing_features = housing.drop('median_house_value',axis=1)
    housing_labels = housing['median_house_value'].copy()

    num_att = housing_features.drop('ocean_proximity',axis=1).columns.tolist()
    cat_att = ['ocean_proximity']
    
    pipeline = build_pipeline(num_att, cat_att)
    housing_prepared = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

    print("Model trained and saved.")

else:
    # Inference Phase
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv('input.csv')
    transform_input = pipeline.transform(input_data)
    predictions = model.predict(transform_input) 
    input_data['median_house_value'] = predictions

    input_data.to_csv("output.csv",index=False)
    print("Inference completed and results saved to output.csv")