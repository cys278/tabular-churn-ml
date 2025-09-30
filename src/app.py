from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Load the tained model
MODEL_PATH="models/churn_model.pkl"
model=joblib.load(MODEL_PATH)

app=FastAPI(title="Churn Prediction API")

#Define input schema
class Customer(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def root():
    return {"message":"Churn Prediction API is running"}

@app.post("/predict")
def predict(customer:Customer):

    # Covert Pydantic object -> dict -> Dataframe to Dataframe
    X=pd.DataFrame([customer.dict()])

    #Predict
    prob=model.predict_proba(X)[0,1]
    label= int(prob>=0.5)

    return {"prob":round(float(prob),3),"label":label}
