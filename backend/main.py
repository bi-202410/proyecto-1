from typing import Optional
from fastapi import FastAPI
from pandas import DataFrame

from data_model import DataModel, TrainModel
from model import Model

app = FastAPI()

DataModelArray = list[DataModel]
TrainModelArray = list[TrainModel]

@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.post("/retrain")
def retrain_model(data: TrainModelArray):
    raw_data = DataFrame([item.model_dump() for item in data], columns=data[0].columns())
    model = Model()
    try:
        res = model.retrain(raw_data)
        return res, 200
    except Exception as e:
        return {"error": str(e)}, 404

@app.post("/predict")
def make_predictions(data: DataModelArray):
    """
    @param data: DataModel representa una instancia de datos a predecir
    """
    #print(data)
    raw_data = DataFrame([item.model_dump() for item in data], columns=data[0].columns())
    #print(raw_data)
    model = Model()
    try:
        prediction = model.make_predictions(raw_data)
        return prediction
    except Exception as e:
        return {"error": str(e)}, 404