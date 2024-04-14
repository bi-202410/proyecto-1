from joblib import load, dump
from pandas import DataFrame

class Model:

    def __init__(self):
        self.model = load("assets/model.joblib")

    def make_predictions(self, data: DataFrame):
        print(f"[INFO] Predicting {data.shape[0]} samples")
        try:
            result = self.model.predict(data)
        except Exception as e:
            print(f"[ERROR] Error while making predictions: {e}")
            raise e
        return result
    
    def retrain(self, data: DataFrame):
        print(f"[INFO] Retraining model based on {data.shape[0]} samples")
        try:
            result = self.model.fit(data)
            dump(result, "assets/model.joblib")
            print(f"[INFO] Model trained successfully and saved to assets/model.joblib")
        except Exception as e:
            print(f"[ERROR] Error while training model: {e}")
            raise e
        return result
