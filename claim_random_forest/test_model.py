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
    risk_score = probability * 100

print("Risk Score:", round(risk_score, 2), "/ 100")
feature_names = [
    "DIAGNOSIS_COUNT",
    "PRVDR_STATE_CD",
    "LINE_PLACE_OF_SRVC_CD",
    "LINE_NUM"
]

for feature, importance in zip(feature_names, model.feature_importances_):
    print(feature, ":", round(importance, 3))
    importances = dict(zip(feature_names, model.feature_importances_))

top_feature = max(importances, key=importances.get)

reason_map = {
    "DIAGNOSIS_COUNT": "Unusual diagnosis pattern",
    "PRVDR_STATE_CD": "Provider location pattern contributed to risk",
    "LINE_PLACE_OF_SRVC_CD": "Unusual place of service pattern",
    "LINE_NUM": "Unusual number of claim service lines"
}
print("Primary Model Driver:", reason_map[top_feature])
