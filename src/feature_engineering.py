import numpy as np
import pandas as pd

def engineer_fraud_features(df):
    """
    Simulates and engineers highly predictive behavioral metrics:
    1. Transaction Velocity (Rolling count of activities per user profile)
    2. Geographic Consistency (Flagging impossibly fast spatial movements)
    """
    print("[*] Humanizing features: Calculating spatial and temporal variations...")
    df = df.copy()
    
    # Sort data by time to simulate a historical sequential ledger
    df = df.sort_values(by='Time').reset_index(drop=True)
    
    # Mocking user tracking profiles based on PCA structural variations 
    # Real-world datasets pivot on actual User/Card IDs; we use V1 variance as a structural proxy
    df['user_proxy_id'] = pd.qcut(df['V1'].rank(method='first'), q=1000, labels=False)
    
    # Feature 1: Transaction Velocity (Rolling 5-transaction window time deltas)
    df['time_delta'] = df.groupby('user_proxy_id')['Time'].diff().fillna(3600)
    df['tx_velocity_score'] = 1.0 / (df['time_delta'] + 1)
    
    # Feature 2: Geographical Consistency (Simulating structural distance anomalies)
    # High amplitude variation in sequential PCA elements implies rapid characteristic shifts
    v2_diff = df.groupby('user_proxy_id')['V2'].diff().fillna(0).abs()
    df['geo_consistency_deviation'] = v2_diff / (df['time_delta'] + 1)
    
    # Drop intermediate processing columns to keep feature space clean
    df = df.drop(columns=['user_proxy_id', 'time_delta'])
    
    print(f"[+] Feature space expanded. Matrix dimensions: {df.shape}")
    return df