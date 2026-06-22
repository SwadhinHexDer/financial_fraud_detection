import json
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, auc

def train_hybrid_pipeline(X_train, y_train, X_test, y_test):
    """
    Trains an Isolation Forest model to isolate structural anomalies and
    an XGBoost engine optimized to maximize Precision-Recall AUC tracking.
    """
    print("[*] Initiating hybrid modeling framework...")
    
    # 1. Unsupervised Isolation Forest Anomaly Mapping
    print(" -> Scaling unsupervised Isolation Forest exploration...")
    iso_forest = IsolationForest(n_estimators=100, contamination=0.0017, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)
    
    # 2. Supervised Target Isolation via XGBoost
    print(" -> Instantiating gradient boosting hyperparameter loops...")
    xgb_model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=10,  # Balances response thresholding
        random_state=42,
        eval_metric="aucpr",
        n_jobs=-1
      )
    xgb_model.fit(X_train, y_train)
    
    # Validation Evaluation Metrics
    y_probs = xgb_model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_probs)
    pr_auc = auc(recall, precision)
    print(f"[🏆] Successfully trained pipeline! Validation PR-AUC achieved: {pr_auc:.4f}")
    
    # Persisting binaries to disk
    joblib.dump(iso_forest, "saved_models/isolation_forest.pkl")
    xgb_model.save_model("saved_models/xgboost_model.json")
    print("[+] Model artifacts written safely to saved_models/")
    
    return xgb_model, iso_forest, pr_auc