from typing import Optional
from fastapi import FastAPI
from pandas import DataFrame

from data_model import DataModel
from model import Model

app = FastAPI()


@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.post("/predict")
def make_predictions(data: DataModel):
    raw_data = DataFrame(data.model_dump(), columns=data.columns(), index=[0])
    model = Model()
    prediction = model.make_predictions(raw_data)
    return prediction[0]

@app.post("/retrain")
def retrain_model(data: Model):
    raw_data = DataFrame(data.model_dump(), columns=data.columns(), index=[0])
    model = Model()
    res = model.retrain(raw_data)
    return res