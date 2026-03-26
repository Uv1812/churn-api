import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

class InputData(BaseModel):
    gender: int                                    # 1=Male 0=Female
    SeniorCitizen: int                             # 1=Yes 0=No
    Partner: int                                   # 1=Yes 0=No
    Dependents: int                                # 1=Yes 0=No
    tenure: int                                    # months 0-72
    PhoneService: int                              # 1=Yes 0=No
    PaperlessBilling: int                          # 1=Yes 0=No
    MonthlyCharges: float                          # e.g. 29.85
    TotalCharges: float                            # e.g. 1889.5
    MultipleLines_No_phone_service: int            # 1 or 0
    MultipleLines_Yes: int                         # 1 or 0
    InternetService_Fiber_optic: int               # 1 or 0
    InternetService_No: int                        # 1 or 0
    OnlineSecurity_No_internet_service: int        # 1 or 0
    OnlineSecurity_Yes: int                        # 1 or 0
    OnlineBackup_No_internet_service: int          # 1 or 0
    OnlineBackup_Yes: int                          # 1 or 0
    DeviceProtection_No_internet_service: int      # 1 or 0
    DeviceProtection_Yes: int                      # 1 or 0
    TechSupport_No_internet_service: int           # 1 or 0
    TechSupport_Yes: int                           # 1 or 0
    StreamingTV_No_internet_service: int           # 1 or 0
    StreamingTV_Yes: int                           # 1 or 0
    StreamingMovies_No_internet_service: int       # 1 or 0
    StreamingMovies_Yes: int                       # 1 or 0
    Contract_One_year: int                         # 1 or 0
    Contract_Two_year: int                         # 1 or 0
    PaymentMethod_Credit_card_automatic: int       # 1 or 0
    PaymentMethod_Electronic_check: int            # 1 or 0
    PaymentMethod_Mailed_check: int                # 1 or 0

@app.post('/predict')
def predict(data:InputData):
    input_dict=data.dict()
    rename_map={
       'MultipleLines_No_phone_service': 'MultipleLines_No phone service',
        'InternetService_Fiber_optic': 'InternetService_Fiber optic',
        'OnlineSecurity_No_internet_service': 'OnlineSecurity_No internet service',
        'OnlineBackup_No_internet_service': 'OnlineBackup_No internet service',
        'DeviceProtection_No_internet_service': 'DeviceProtection_No internet service',
        'TechSupport_No_internet_service': 'TechSupport_No internet service',
        'StreamingTV_No_internet_service': 'StreamingTV_No internet service',
        'StreamingMovies_No_internet_service': 'StreamingMovies_No internet service',
        'Contract_One_year': 'Contract_One year',
        'Contract_Two_year': 'Contract_Two year',
        'PaymentMethod_Credit_card_automatic': 'PaymentMethod_Credit card (automatic)',  
        'PaymentMethod_Electronic_check': 'PaymentMethod_Electronic check',  
        'PaymentMethod_Mailed_check': 'PaymentMethod_Mailed check'
    }
    
    for old_name, new_name in rename_map.items():
        input_dict[new_name] = input_dict.pop(old_name)

    feature_order = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges',
        'MultipleLines_No phone service', 'MultipleLines_Yes',
        'InternetService_Fiber optic', 'InternetService_No',
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
        'OnlineBackup_No internet service', 'OnlineBackup_Yes',
        'DeviceProtection_No internet service', 'DeviceProtection_Yes',
        'TechSupport_No internet service', 'TechSupport_Yes',
        'StreamingTV_No internet service', 'StreamingTV_Yes',
        'StreamingMovies_No internet service', 'StreamingMovies_Yes',
        'Contract_One year', 'Contract_Two year',
        'PaymentMethod_Credit card (automatic)',
        'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]

    input_df=pd.DataFrame([input_dict])[feature_order]
    input_scaled=scaler.transform(input_df)
    probability=model.predict_proba(input_scaled)[0][1]
    prediction="will churn" if probability>0.5 else "will not churn"

    return{
        "prediction": prediction,
        "churn_probability": round(float(probability),3),
        "confidence":f"{round(float(probability)*100,1)}%"
    }

@app.get('/health')
def health():
    return {'status': 'ok'}