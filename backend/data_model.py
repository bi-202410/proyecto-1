from pydantic import BaseModel
from typing import List
class DataModel(BaseModel):
    review: str
    def columns(self):
        """
        @return list: lista de columnas correspondientes al modelo exportado
        """
        return ["review"]
class TrainModel(BaseModel):
    review: str
    class_: int
    def columns(self):
        """
        @return list: lista de columnas correspondientes al modelo exportado
        """
        return ["review","class_"]
class PredictionResult(BaseModel):
    review: str
    prediction: List
