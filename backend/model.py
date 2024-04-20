# import necessary libraries
import numpy as np
import pandas as pd
from pandas import DataFrame

# import sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score

#Pipeline
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

#preprocesing
from preprocessing import tokenize_text, train_evaluate_pipeline
from data_model import PredictionResult
class Model:
    def __init__(self):
        self.model = load("assets/pipeline.joblib")

    def retrain(self, data: DataFrame):
        print(f"[INFO] Retraining model based on {data.shape[0]} samples")
        try:
            #print(data)
            df_prep = data.drop_duplicates()
            X_train, X_test, y_train, y_test = train_test_split(df_prep["review"], df_prep["class_"], test_size=0.3, stratify=df_prep["class_"], random_state=42)
            pipeline = self.model
            evaluation_metrics = train_evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test)
            print(f"[INFO] Model trained successfully and saved to assets/model.joblib")
        except Exception as e:
            print(f"[ERROR] Error while training model: {e}")
            raise e
        return evaluation_metrics

    def make_predictions(self, data: DataFrame):
        print(f"[INFO] Predicting {data.shape[0]} samples")
        predictions = []
        model = Model()
        for index, row in data.iterrows():
            review = row["review"]
            try:
                probs = self.model.predict_proba([review])
                prediction = [list(prob) for prob in probs]
                prediction_result = PredictionResult(review=review, prediction=prediction)
                predictions.append(prediction_result)
            except Exception as e:
                print(f"[ERROR] Error while making predictions: {e}")
                raise e
        return predictions
