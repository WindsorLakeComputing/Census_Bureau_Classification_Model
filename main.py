from fastapi import FastAPI
import joblib
import pandas as pd
import os
from starter.starter.ml.model import inference
from starter.starter.ml.data import process_data, get_cat_features

from pydantic import BaseModel

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")


def replace_hyphens(string: str) -> str:
    return string.replace('_','-')

class CensusEntry(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        alias_generator = replace_hyphens



app = FastAPI()

@app.on_event("startup")
async def startup_event():
  global model, encoder, binarizer
  model = joblib.load("./starter/model/lgbm_class.pkl")
  encoder = joblib.load("./starter/model/encoder.pkl")
  binarizer = joblib.load("./starter/model/lb.pkl")

@app.get("/")
async def say_hello():
    return {"greeting": "Hello there!"}

@app.post("/census/")
async def create_item(entry: CensusEntry):
    entry_underscores = entry.dict(by_alias=True)
    df = pd.DataFrame(entry_underscores, index=[0])
    df = df.rename(columns={'marital_status': 'marital-status', 'native_country': 'native-country'})
    X_test, y_test, new_encoder, new_lib = process_data(
        df, categorical_features=get_cat_features(), label=None, training=False, encoder=encoder, lb=binarizer
    )
    preds = inference(model, X_test)
    answer = ""
    if(preds[0] == 0):
        answer = "<=50K"
    else:
        answer = ">50K"
    return ("The prediction is that the salary is " + answer)