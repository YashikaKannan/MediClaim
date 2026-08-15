import joblib
import os

model_path = os.path.join(
    os.path.dirname(__file__),
    "random_forest_claim_model.pkl"
)

model = joblib.load(model_path)

print("Random Forest model loaded successfully!")

import pandas as pd

sample_claim = pd.DataFrame([{
    "DIAGNOSIS_COUNT": 10,
    "PRVDR_STATE_CD": 4,
    "LINE_PLACE_OF_SRVC_CD": 11,
    "LINE_NUM": 12
}])

probability = model.predict_proba(sample_claim)[0][1]

print("Suspicious probability:", probability)

threshold = 0.8

if probability >= threshold:
    print("Prediction: Suspicious Claim")
else:
    print("Prediction: Normal Claim")