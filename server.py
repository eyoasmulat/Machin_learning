import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Load the trained model and scaler
model = joblib.load('model_nn.joblib')
# scaler = joblib.load('scaler.joblib')  # Uncomment if you have a scaler

# Define the input data schema using Pydantic
class InjuryPredictionInput(BaseModel):
    footballer_id: int
    postion: str
    value: float
    game_workload: int
    year: int
    month: int
    day: int

# Initialize FastAPI app
app = FastAPI()

# Define the prediction endpoint
@app.post("/predict")
def predict_injury(input_data: InjuryPredictionInput):
    try:
        # Convert input data to a DataFrame
        input_dict = input_data.dict()
        input_df = pd.DataFrame([input_dict])

        # Encode the 'postion' column (if necessary)
        input_df['postion'] = input_df['postion'].map({'midfilder': 0, 'attacker': 1})

        # Standardize the features (if you have a scaler)
        # input_scaled = scaler.transform(input_df)

        # Make a prediction
        prediction = model.predict(input_df)  # Use input_scaled if you have a scaler
        prediction_proba = model.predict_proba(input_df)  # Use input_scaled if you have a scaler

        # Return the prediction result
        return {
            "prediction": int(prediction[0]),
            "probability": float(prediction_proba[0][1]),
            "message": "Injury predicted successfully."
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Sports Injury Prediction API!"}