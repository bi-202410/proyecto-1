from joblib import load, dump
from pandas import DataFrame

class Model:

    def __init__(self, columns):
        self.model = load("assets/model.joblib")

    def make_predictions(self, data: DataFrame):
        result = self.model.predict(data)
        return result
    
    def retrain(self, data: DataFrame):
        result = self.model.fit(data)
        dump(result, "assets/model.joblib")
        return result
