import sqlite3
import os
import pandas as pd
from xgboost import XGBClassifier

DB_ROOT = "Gold_D1_Merged.db"
DB_API = os.path.join("api", "Gold_D1_Merged.db")
FEATURES = ['MA10', 'MA30', 'MA50', 'RSI14', 'MACD', 'Volatility']

def precalculate_and_save(db_path):
    print(f"Processing database: {db_path}")
    if not os.path.exists(db_path):
        print(f"Error: Database {db_path} not found.")
        return

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM gold_analytics ORDER BY Date ASC", conn)
    
    if df.empty:
        print("Error: gold_analytics table is empty.")
        conn.close()
        return

    # Define target: 1 if tomorrow's Close > today's Close, else 0
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Train the XGBoost model on the first 70% of chronological data
    clean_df = df.dropna(subset=FEATURES + ['Target'])
    X = clean_df[FEATURES]
    y = clean_df['Target']
    
    split_idx = int(len(clean_df) * 0.7)
    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]
    
    print("Training XGBoost model...")
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
    print("Model trained successfully!")

    # Create predictions table
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS gold_predictions (
            Date TEXT PRIMARY KEY,
            Prediction TEXT,
            Prob_Up REAL,
            Prob_Down REAL
        )
    """)
    conn.commit()

    # Predict for all rows that have clean features
    predict_df = df.dropna(subset=FEATURES)
    X_pred = predict_df[FEATURES]
    
    preds = model.predict(X_pred)
    probas = model.predict_proba(X_pred)

    records = []
    for idx, row in predict_df.iterrows():
        date = row['Date']
        pred_val = int(preds[predict_df.index.get_loc(idx)])
        proba = probas[predict_df.index.get_loc(idx)]
        
        prediction = "TĂNG" if pred_val == 1 else "GIẢM"
        prob_down = float(proba[0])
        prob_up = float(proba[1])
        
        records.append((date, prediction, round(prob_up * 100, 2), round(prob_down * 100, 2)))

    # Insert into the database
    cursor.executemany("""
        INSERT OR REPLACE INTO gold_predictions (Date, Prediction, Prob_Up, Prob_Down)
        VALUES (?, ?, ?, ?)
    """, records)
    conn.commit()
    
    # Verify count
    cursor.execute("SELECT COUNT(*) FROM gold_predictions")
    count = cursor.fetchone()[0]
    print(f"Successfully calculated and saved {count} predictions to {db_path}!")
    conn.close()

if __name__ == "__main__":
    precalculate_and_save(DB_ROOT)
    precalculate_and_save(DB_API)
