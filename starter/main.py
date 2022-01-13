from fastapi import FastAPI
import joblib
import pandas as pd
import json
from starter.ml.model import inference
from starter.ml.data import process_data

from pydantic import BaseModel

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

# Instantiate the app.
app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/census/")
async def create_item(entry: CensusEntry):
    lgbm_class = joblib.load('./model/lgbm_class.pkl')
    encoder = joblib.load('./model/encoder.pkl')
    lb = joblib.load('./model/lb.pkl')
    print("entry is")
    print(entry.__dict__)
    dict = entry.__dict__
    #df = pd.DataFrame.from_dict(dict)
    df = pd.DataFrame([dict])
    print('dict[race] is')
    print(dict['race'])
    jsonStr = json.dumps(entry.__dict__)
    #df = pd.read_json(jsonStr, orient='index')
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    #print("df.head()")
    #print(df.head())
    df = df.rename(columns={'marital_status': 'marital-status', 'native_country': 'native-country'})
    print(df['marital-status'])
    X_test, y_test, new_encoder, new_lib = process_data(
        df, categorical_features=cat_features, label=None, training=False, encoder=encoder, lb=lb
    )
    preds = inference(lgbm_class, X_test)
    print('preds is ')
    print(preds)



    return entry
