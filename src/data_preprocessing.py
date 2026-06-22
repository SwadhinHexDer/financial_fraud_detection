import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def pipeline_preprocessing(data_path, target_col='Class'):
    """
    Loads dataset, scales continuous features, performs a clean historical split,
    and isolates minority classes using SMOTE purely inside the training boundary.
    """
    print(f"[*] Processing dataset located at: {data_path}")
    df = pd.read_csv(data_path)
    
    # Scale raw un-normalized features
    scaler = StandardScaler()
    df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
    df = df.drop(columns=['Amount'])
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Strategic train/test split to secure valid validation sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"[!] Target imbalance before execution: {dict(y_train.value_counts())}")
    
    # Apply SMOTE exclusively on training splits to prevent predictive leakage
    smote = SMOTE(sampling_strategy=0.10, random_state=42) 
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    print(f"[+] Post-SMOTE training matrix balanced: {dict(y_train_res.value_counts())}")
    
    return X_train_res, X_test, y_train_res, y_test, scaler