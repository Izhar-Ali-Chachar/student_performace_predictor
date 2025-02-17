"""
Serve PyCaret prediction pipeline using FastAPI.
"""
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

app = FastAPI()

model = load_model("best_model")

@app.post("/predict")
def predict(
        gender: str,
        race: str,
        pEducation: str,
        lunch: str,
        tpcourse: str,
        rscore: int,
        wscore: int
):
    data = pd.DataFrame([[gender, race, pEducation, lunch, tpcourse, rscore, wscore]])
    data.columns = [
        'gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'reading score',
        'writing score'
    ]

    predictions = predict_model(model, data=data)
    return {"prediction": int(predictions["prediction_label"][0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)