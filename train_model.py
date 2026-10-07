import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, classification_report

print("=== SafeRoute AI: Step 1 - Model Training Pipeline ===")

# 1. Load Master Dataset
data_path = "safe_route_master_dataset.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Cannot find {data_path}. Ensure data generation script has completed.")

print(f"Loading {data_path}...")
df = pd.read_csv(data_path, encoding="utf-8")
print(f"Loaded {len(df)} road segments.")

# 2. Select Features (Excluding derived targets to avoid data leakage)
feature_cols = [
    "highway_type", 
    "length_meters", 
    "oneway", 
    "lighting_status", 
    "lighting_probability",
    "police_dist_m", 
    "hospital_dist_m", 
    "traffic_volume", 
    "traffic_speed",
    "lane_occupancy_pct", 
    "congestion_delay_sec", 
    "historical_crime_severity",
    "historical_violent_ratio", 
    "expected_police_response"
]

X_raw = df[feature_cols].copy()
y_reg = df["risk_score"]
y_clf = df["risk_class"]

# 3. One-Hot Encode Categorical Features
X = pd.get_dummies(X_raw, columns=["highway_type", "lighting_status"], drop_first=True)

# Save feature column blueprint for runtime routing inference
model_columns = list(X.columns)
joblib.dump(model_columns, "model_features.pkl")
print(f"Engineered {len(model_columns)} input feature columns.")

# 4. Train Risk Regressor (Used directly by routing graph cost functions)
print("\n--- Training Random Forest Regressor ---")
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X, y_reg, test_size=0.20, random_state=42
)

regressor = RandomForestRegressor(
    n_estimators=100, 
    max_depth=15, 
    random_state=42, 
    n_jobs=-1
)
regressor.fit(X_train_r, y_train_r)

r_preds = regressor.predict(X_test_r)
r2 = r2_score(y_test_r, r_preds)
rmse = np.sqrt(mean_squared_error(y_test_r, r_preds))
print(f"Regressor Performance -> R² Score: {r2:.4f} | RMSE: {rmse:.4f}")

joblib.dump(regressor, "risk_model_regressor.pkl")
print("Saved regressor to 'risk_model_regressor.pkl'")

# 5. Train Risk Classifier (Used for UI alerts and map popups)
print("\n--- Training Random Forest Classifier ---")
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)

classifier = RandomForestClassifier(
    n_estimators=100, 
    max_depth=15, 
    random_state=42, 
    n_jobs=-1
)
classifier.fit(X_train_c, y_train_c)

c_preds = classifier.predict(X_test_c)
print("Classifier Classification Report:")
print(classification_report(y_test_c, c_preds))

joblib.dump(classifier, "risk_model_classifier.pkl")
print("Saved classifier to 'risk_model_classifier.pkl'")

# 6. Feature Importance Inspection
importances = pd.Series(regressor.feature_importances_, index=model_columns).sort_values(ascending=False)
print("\nTop 5 Most Influential Features on Road Risk:")
print(importances.head(5).round(4))

print("\nModel Training Complete! Artifacts generated: 'risk_model_regressor.pkl', 'risk_model_classifier.pkl', 'model_features.pkl'")