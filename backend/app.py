from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib, pandas as pd

app = FastAPI(title="Student Depression Risk API")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

model = joblib.load("model.joblib")

class Student(BaseModel):
    gender: str
    age: float = Field(ge=15, le=60)
    academic_pressure: float = Field(ge=1, le=5)
    cgpa: float = Field(ge=0, le=10)
    study_satisfaction: float = Field(ge=1, le=5)
    sleep_duration: str
    dietary_habits: str
    work_study_hours: float = Field(ge=0, le=12)
    financial_stress: float = Field(ge=1, le=5)
    family_history_of_mental_illness: str

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(s: Student):
    row = pd.DataFrame([s.model_dump()])
    p = float(model.predict_proba(row)[0][1])
    level = "High" if p >= 0.7 else "Moderate" if p >= 0.4 else "Low"
    return {"probability": round(p * 100, 1), "risk_level": level}