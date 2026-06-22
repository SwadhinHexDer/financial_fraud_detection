# run_pipeline.py
from src.data_preprocessing import pipeline_preprocessing
from src.feature_engineering import engineer_fraud_features
from src.model import train_hybrid_pipeline
import pandas as pd
import os

print("=== STARTING FRAUD DETECTION SYSTEM PIPELINE ===")
# 1. Feature Engineering
raw_df = pd.read_csv('data/creditcard.csv')
enriched_df = engineer_fraud_features(raw_df)
enriched_df.to_csv('data/enriched_creditcard.csv', index=False)

# 2. Split and Preprocess with SMOTE
X_train, X_test, y_train, y_test, scaler = pipeline_preprocessing('data/enriched_creditcard.csv')

# 3. Train and Save
xgb, iso, pr_auc = train_hybrid_pipeline(X_train, y_train, X_test, y_test)
print("=== PIPELINE COMPLETION SUCCESSFUL ===")