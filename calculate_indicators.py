import sqlite3
import pandas as pd
import numpy as np

def calculate_technical_indicators(db_filepath):
    print("Connecting to database...")
    conn = sqlite3.connect(db_filepath)
    
    # 1. Load data from SQLite database
    # Sorting by Date ascending is critical for time-series rolling calculations!
    df = pd.read_sql_query("SELECT Date, Close, High, Low, Open, Volume, Open_interest FROM gold_d1 ORDER BY Date ASC", conn)
    print(f"Successfully loaded {len(df)} rows.")
    
    # --- TECHNICAL INDICATOR CALCULATIONS ---
    
    # A. Simple Moving Averages (MA10, MA30, MA50)
    # Công thức: MA_N = (Close_t + Close_{t-1} + ... + Close_{t-N+1}) / N
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA30'] = df['Close'].rolling(window=30).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # B. Relative Strength Index (RSI14)
    # Công thức: 
    #   1. Tính mức tăng/giảm hàng ngày: Diff = Close_t - Close_{t-1}
    #   2. Phân tách thành Gain (nếu Diff > 0) và Loss (nếu Diff < 0, lấy trị tuyệt đối)
    #   3. Tính trung bình chuyển mũ (Wilder's EMA) 14 phiên cho Gain và Loss:
    #      AvgGain_t = (AvgGain_{t-1} * 13 + Gain_t) / 14
    #      AvgLoss_t = (AvgLoss_{t-1} * 13 + Loss_t) / 14
    #   4. RS = AvgGain / AvgLoss
    #   5. RSI = 100 - [100 / (1 + RS)]
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    # Wilder's Exponential Moving Average smoothing method (com=13 corresponds to span=27, alpha=1/14)
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + rs))
    # Fill NaN for first row which has no difference
    df.loc[0, 'RSI14'] = np.nan 
    
    # C. Moving Average Convergence Divergence (MACD)
    # Công thức chuẩn (12, 26, 9):
    #   1. EMA_12 = Close_t * k_12 + EMA_{t-1} * (1 - k_12)  với k_12 = 2/(12+1)
    #   2. EMA_26 = Close_t * k_26 + EMA_{t-1} * (1 - k_26)  với k_26 = 2/(26+1)
    #   3. Đường MACD (MACD_Line) = EMA_12 - EMA_26
    #   4. Đường Tín hiệu (Signal_Line) = EMA 9 phiên của Đường MACD: MACD_Line * k_9 + Signal_{t-1} * (1 - k_9) với k_9 = 2/(9+1)
    #   5. Biểu đồ MACD (MACD_Hist) = MACD_Line - Signal_Line
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # D. Historical Volatility (Độ biến động lịch sử - 20 phiên)
    # Công thức:
    #   1. Tính tỷ suất sinh lời hàng ngày (Daily Returns): R_t = (Close_t - Close_{t-1}) / Close_{t-1}
    #   2. Độ biến động hàng ngày = Độ lệch chuẩn (Standard Deviation) cuốn chiếu 20 phiên của R_t
    #   3. Để chuẩn hóa theo tỷ lệ phần trăm dễ đọc, chúng ta sẽ nhân kết quả với 100 (%)
    #   (Có thể nhân thêm sqrt(252) nếu muốn tính độ biến động theo năm, ở đây tính theo ngày để bám sát dữ liệu D1)
    daily_returns = df['Close'].pct_change()
    df['Volatility'] = daily_returns.rolling(window=20).std() * 100  # Đơn vị: %
    
    # --- ROUND VALUES FOR CLEAN STORAGE ---
    df['MA10'] = df['MA10'].round(2)
    df['MA30'] = df['MA30'].round(2)
    df['MA50'] = df['MA50'].round(2)
    df['RSI14'] = df['RSI14'].round(2)
    df['MACD'] = df['MACD'].round(4)
    df['MACD_Signal'] = df['MACD_Signal'].round(4)
    df['MACD_Hist'] = df['MACD_Hist'].round(4)
    df['Volatility'] = df['Volatility'].round(4)
    
    # --- SAVE TO NEW DATABASE TABLE 'gold_analytics' ---
    print("Writing calculated technical indicators back to database...")
    
    # Drop table if exists to update fresh values
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS gold_analytics")
    conn.commit()
    
    # Use pandas to_sql to export the calculated dataframe into a new table
    # This automatically maps columns to SQLite data types
    df.to_sql('gold_analytics', conn, index=False, if_exists='replace')
    
    # Create Index on Date to make future queries extremely fast
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_date ON gold_analytics (Date)")
    conn.commit()
    
    # --- VERIFY AND PRINT SAMPLES ---
    print("\nCalculations completed successfully!")
    print("--- SAMPLE OF RECENT DAYS CALCULATED (Last 5 records) ---")
    latest_data = df.tail(5)
    for idx, row in latest_data.iterrows():
        print(f"Date: {row['Date']}")
        print(f"  Close: {row['Close']:.2f} USD")
        print(f"  MA10: {row['MA10']:.2f} | MA30: {row['MA30']:.2f} | MA50: {row['MA50']:.2f}")
        print(f"  RSI14: {row['RSI14']:.2f}% (Relative Strength Index)")
        print(f"  MACD Line: {row['MACD']:.4f} | Signal: {row['MACD_Signal']:.4f} | Hist: {row['MACD_Hist']:.4f}")
        print(f"  Volatility (20d): {row['Volatility']:.4f}%")
        print("-" * 50)
        
    conn.close()
    print("Database connection closed. Table 'gold_analytics' is ready for use.")

if __name__ == "__main__":
    db_file = "Gold_D1_Merged.db"
    calculate_technical_indicators(db_file)
