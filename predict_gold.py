import sqlite3
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import os

def run_xgboost_gold_prediction(db_filepath, train_ratio=0.7):
    print("--- XGBOOST GOLD PRICE PREDICTION TERMINAL ---")
    print(f"Connecting to database: {db_filepath}")
    
    # 1. Load the technical indicators calculated previously
    conn = sqlite3.connect(db_filepath)
    df = pd.read_sql_query("SELECT * FROM gold_analytics ORDER BY Date ASC", conn)
    conn.close()
    
    # 2. Define features and target
    # Features requested
    features = ['MA10', 'MA30', 'MA50', 'RSI14', 'MACD', 'Volatility']
    
    # Target: 1 (Tăng) if tomorrow's Close > today's Close, 0 (Giảm) otherwise.
    # We shift Close back by -1 to compare tomorrow's Close with today's Close.
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # The last row represents today (the latest date in database). 
    # Its target is NaN because we don't know tomorrow's Close price yet.
    # We isolate this last row for predicting tomorrow's price direction out-of-sample!
    latest_row = df.iloc[-1:]
    
    # 3. Clean up data: Remove rows where indicators are NaN (first 49 rows due to MA50) 
    # and drop the last row from the training dataset.
    clean_df = df.dropna(subset=features + ['Target'])
    
    X = clean_df[features]
    y = clean_df['Target']
    
    # 4. Split into Train and Test using 3:7 split ratio (70% Train, 30% Test)
    # Using shuffle=False is CRITICAL for financial time-series to prevent look-ahead bias!
    test_ratio = 1 - train_ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_ratio, 
        shuffle=False
    )
    
    train_size = len(X_train)
    test_size = len(X_test)
    
    print(f"Total historical samples: {len(clean_df)}")
    print(f"  Training Set ({train_ratio*100:.0f}%): {train_size} rows (from {clean_df.iloc[0]['Date']} to {clean_df.iloc[train_size-1]['Date']})")
    print(f"  Testing Set ({test_ratio*100:.0f}%): {test_size} rows (from {clean_df.iloc[train_size]['Date']} to {clean_df.iloc[-1]['Date']})")
    
    # 5. Initialize and Train XGBoost Classifier
    # Use conservative parameters to avoid overfitting on market noise
    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # 6. Evaluate Model Performance on Test Set (Out-of-sample)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print("\n--- MODEL PERFORMANCE METRICS (ON TEST SET) ---")
    print(f"Accuracy (Forecast Accuracy): {accuracy*100:.2f}%")
    print(f"ROC AUC (UP/DOWN Discrimination): {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['DOWN (0)', 'UP (1)']))
    
    # 7. Predict Tomorrow's Price (Out-of-sample forecast for the latest date)
    latest_date = latest_row['Date'].values[0]
    X_forecast = latest_row[features]
    
    tomorrow_pred = model.predict(X_forecast)[0]
    tomorrow_proba = model.predict_proba(X_forecast)[0]
    
    prob_down = tomorrow_proba[0] * 100
    prob_up = tomorrow_proba[1] * 100
    
    print("==================================================")
    print(f"FORECAST FOR NEXT TRADING DAY (After {latest_date}):")
    print(f"Today's indicators ({latest_date}):")
    print(f"  Close Price: {latest_row['Close'].values[0]:.2f} USD")
    for f in features:
        print(f"  Indicator {f}: {latest_row[f].values[0]:.4f}")
        
    print("\n=== NEXT DAY TREND FORECAST RESULT ===")
    if tomorrow_pred == 1:
        print(f"Result Trend Forecast: UP (INCREASE)")
        print(f"  Probability of UP (Xac suat Tang): {prob_up:.2f}%")
        print(f"  Probability of DOWN (Xac suat Giam): {prob_down:.2f}%")
    else:
        print(f"Result Trend Forecast: DOWN (DECREASE)")
        print(f"  Probability of DOWN (Xac suat Giam): {prob_down:.2f}%")
        print(f"  Probability of UP (Xac suat Tang): {prob_up:.2f}%")
    print("==================================================")

if __name__ == "__main__":
    db_file = "Gold_D1_Merged.db"
    
    # Tỉ lệ học 3:7 (ở đây có nghĩa là 70% Train, 30% Test là tỷ lệ tối ưu nhất)
    # Chúng tôi áp dụng tỷ lệ chia: 70% dữ liệu để Train (học) và 30% để Test (kiểm thử)
    run_xgboost_gold_prediction(db_file, train_ratio=0.7)
