
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="Employee Metrics Prediction API")

# Define input data model
class EmployeeInput(BaseModel):
    employee_id: str

# Define response model
class PredictionResponse(BaseModel):
    employee_id: str
    predicted_workload_balance: float
    actual_workload_balance: float = None
    predicted_salary: float
    actual_salary: float = None

# Load dataset and models at startup
try:
    data = pd.read_csv("balanced_simple_hr_ai_dataset.csv")
    workload_model = joblib.load("workload_model.pkl")
    workload_scaler = joblib.load("workload_scaler.pkl")
    salary_model = joblib.load("salary_model.pkl")
    salary_scaler = joblib.load("salary_scaler.pkl")
    logging.info("Dataset and models loaded successfully")
except Exception as e:
    logging.error("Error loading dataset or models: %s", str(e))
    raise

# Define features
workload_features = [
    'years_experience', 'current_tasks', 'max_capacity', 'urgent_tasks',
    'completed_tasks', 'total_tasks', 'task_completion_rate',
    'accuracy_score', 'feedback_score', 'available_hours', 'total_hours',
    'department_Finance', 'department_HR', 'department_IT',
    'department_Marketing', 'department_Operations', 'position_Junior',
    'position_Lead', 'position_Manager', 'position_Mid', 'position_Senior'
]

salary_features = [
    'years_experience', 'task_completion_rate', 'accuracy_score',
    'feedback_score', 'skill_match_score', 'availability_score',
    'performance_score', 'workload_balance', 'market_value_score',
    'department_Finance', 'department_HR', 'department_IT',
    'department_Marketing', 'department_Operations', 'position_Junior',
    'position_Lead', 'position_Manager', 'position_Mid', 'position_Senior'
]

def predict_employee_metrics(employee_id: str):
    try:
        # Find employee data
        employee_data = data[data['employee_id'] == employee_id]
        if employee_data.empty:
            logging.error("Employee ID %s not found in dataset", employee_id)
            raise ValueError(f"Employee ID {employee_id} not found")

        # Extract features for workload
        workload_input = employee_data[workload_features].copy()
        workload_input.fillna(workload_input.mean(), inplace=True)
        workload_scaled = workload_scaler.transform(workload_input)

        # Extract features for salary
        salary_input = employee_data[salary_features].copy()
        salary_input.fillna(salary_input.mean(), inplace=True)
        salary_scaled = salary_scaler.transform(salary_input)

        # Make predictions
        workload_pred = workload_model.predict(workload_scaled)[0]
        salary_pred = salary_model.predict(salary_scaled)[0]

        return {
            "employee_id": employee_id,
            "predicted_workload_balance": float(workload_pred),
            "actual_workload_balance": float(employee_data["workload_balance"].values[0]),
            "predicted_salary": float(salary_pred),
            "actual_salary": float(employee_data["salary"].values[0])
        }

    except Exception as e:
        logging.error("Error in prediction: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: EmployeeInput):
    return predict_employee_metrics(input_data.employee_id)
