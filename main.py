import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("titanic_model_pipeline.pkl")

# Define input format using Pydantic
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float
    Embarked: str

# Initialize app
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Titanic survival prediction API is running! Use http://127.0.0.1:8000/docs"}

@app.post("/predict")
def predict_survival(passenger: Passenger):
    # Convert input to DataFrame
    input_df = pd.DataFrame([passenger.dict()])
    
    # Predict using loaded pipeline
    prediction = model.predict(input_df)[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"
    
    return {"prediction": result}